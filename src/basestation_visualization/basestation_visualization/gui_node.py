#!/usr/bin/env python3

import sys
import time
import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CompressedImage, PointCloud2, Imu, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2

class BasestationNode(Node):
    def __init__(self):
        super().__init__('basestation_gui_node')
        self.bridge = CvBridge()
        self.state_lock = threading.Lock()
        self.latest_cv_image = None
        self.latest_lidar_image = None
        self.mission_status = 'Waiting for robot'
        self.last_command_time = float('-inf')
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.imu = None
        self.last_imu_time = float('-inf')
        self.camera_info = None
        self.detection_rays = None
        self.last_rays_time = float('-inf')
        self.detected_center = None
        self.detected_contour = None
        self.detected_pose = None
        self.last_detection_time = float('-inf')

        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.image_callback, video_qos)
        self.create_subscription(Twist, '/cmd_vel', self.telemetry_callback, 10)
        self.create_subscription(String, '/mission_control', self.status_callback, 10)
        self.create_subscription(PointCloud2, '/reduced_lidar_points', self.lidar_callback, video_qos)
        self.create_subscription(Imu, '/camera/camera/imu', self.imu_callback, video_qos)
        
        # Small metadata messages annotate the existing compressed video locally.
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, video_qos)
        self.create_subscription(MarkerArray, '/perception/markers', self.detection_markers_callback, video_qos)
        self.create_subscription(PoseStamped, '/perception/box_pose', self.detection_pose_callback, video_qos)

        # Publisher for Mission Control
        self.mission_pub = self.create_publisher(String, '/mission_control', 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"Camera decode failed: {e}")
            return
        try:
            self.draw_detection_overlay(frame, msg.header)
        except Exception as e:
            # Perception metadata must never block otherwise valid video.
            self.get_logger().warning(f"Detection overlay failed: {e}")
        with self.state_lock:
            self.latest_cv_image = frame

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def detection_markers_callback(self, msg):
        for marker in msg.markers:
            if marker.ns == 'detection_rays':
                self.detection_rays = marker if marker.action == Marker.ADD else None
                self.last_rays_time = time.monotonic()
                continue
            if marker.action != Marker.ADD:
                continue
            if marker.ns == 'box_center_camera':
                self.detected_center = marker
            elif marker.ns == 'detected_contour':
                self.detected_contour = marker

    def detection_pose_callback(self, msg):
        self.detected_pose = msg
        self.last_detection_time = time.monotonic()

    @staticmethod
    def stamp_seconds(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def draw_detection_overlay(self, frame, image_header):
        """Overlay fresh perception metadata on the original camera stream."""
        pose, point, info = self.detected_pose, self.detected_center, self.camera_info
        fresh_pose = (pose is not None and time.monotonic()-self.last_detection_time < 0.6
                      and abs(self.stamp_seconds(pose.header)-self.stamp_seconds(image_header)) < 0.5)
        pose_drawn = False
        lines = ['Object: unavailable / stale']
        if fresh_pose:
            position, q = pose.pose.position, pose.pose.orientation
            xyz = np.array([position.x, position.y, position.z])
            quat = np.array([q.x, q.y, q.z, q.w])
            valid_pose = np.isfinite(xyz).all() and np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-8
            if valid_pose:
                x, y, z, w = quat / np.linalg.norm(quat)
                roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
                pitch = math.asin(float(np.clip(2*(w*y-z*x), -1, 1)))
                yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                rpy = np.degrees([roll, pitch, yaw])
                lines = [f'Box center distance ({pose.header.frame_id}): {np.linalg.norm(xyz):.2f} m',
                         f'XYZ: {xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f} m',
                         f'RPY: {rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f} deg']
                # Project the same box center used by the distance report, plus its contour.
                if (point is not None and info is not None
                        and point.header.frame_id == image_header.frame_id == info.header.frame_id
                        and abs(self.stamp_seconds(point.header)-self.stamp_seconds(pose.header)) < 0.05
                        and (info.width, info.height) == (frame.shape[1], frame.shape[0])):
                    center = point.pose.position
                    xyz_camera = np.array([center.x, center.y, center.z], np.float64)
                    k = np.asarray(info.k, np.float64).reshape(3, 3)
                    distortion = np.asarray(info.d, np.float64)
                    if (info.distortion_model in ('plumb_bob', 'rational_polynomial')
                            and np.isfinite(xyz_camera).all() and xyz_camera[2] > 0
                            and np.isfinite(k).all() and np.isfinite(distortion).all()
                            and k[0, 0] > 0 and k[1, 1] > 0):
                        pixel, _ = cv2.projectPoints(xyz_camera[None], np.zeros(3), np.zeros(3), k, distortion)
                        uv = pixel.ravel()
                        if np.isfinite(uv).all() and 0 <= uv[0] < frame.shape[1] and 0 <= uv[1] < frame.shape[0]:
                            u, v = np.rint(uv).astype(int)
                            contour = self.detected_contour
                            if (contour is not None and contour.header.frame_id == point.header.frame_id
                                    and abs(self.stamp_seconds(contour.header)-self.stamp_seconds(point.header)) < 0.001
                                    and len(contour.points) >= 4):
                                corners = np.array([[p.x, p.y, p.z] for p in contour.points], np.float64)
                                if np.isfinite(corners).all() and np.all(corners[:, 2] > 0):
                                    outline, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), k, distortion)
                                    if np.isfinite(outline).all():
                                        outline = np.clip(outline.reshape(-1, 2), -10000, 10000).astype(np.int32)
                                        cv2.polylines(frame, [outline], True, (0, 255, 0), 2)
                            cv2.circle(frame, (u, v), 4, (0, 255, 255), -1)
                            pose_drawn = True
        rays = self.detection_rays
        if (not pose_drawn and rays is not None and info is not None and len(rays.points) == 10
                and time.monotonic()-self.last_rays_time < 0.6
                and rays.header.frame_id == image_header.frame_id == info.header.frame_id
                and abs(self.stamp_seconds(rays.header)-self.stamp_seconds(image_header)) < 0.5
                and (info.width, info.height) == (frame.shape[1], frame.shape[0])):
            endpoints = np.array([[p.x,p.y,p.z] for p in rays.points[1::2]], np.float64)
            if np.isfinite(endpoints).all() and np.all(endpoints[:,2] > 0):
                pixels, _ = cv2.projectPoints(endpoints, np.zeros(3), np.zeros(3),
                                             np.asarray(info.k,np.float64).reshape(3,3), np.asarray(info.d,np.float64))
                if np.isfinite(pixels).all():
                    pixels = np.clip(pixels.reshape(-1,2), -10000,10000).astype(np.int32)
                    cv2.polylines(frame, [pixels[:4]], True, (0,255,0), 2)
                    # A depth-based box center replaces this centroid when a current pose exists.
                    if not fresh_pose:
                        cv2.circle(frame, tuple(pixels[4]), 4, (0,255,255), -1)
                        lines = ['Brown target detected', 'Distance / pose unavailable (depth or TF)']
        scale, thickness = 0.48, 1
        widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines]
        panel_width = min(frame.shape[1]-1, max(widths)+20)
        cv2.rectangle(frame, (5, 5), (panel_width, 12+22*len(lines)), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (10, 26+22*index), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def status_callback(self, msg):
        if msg.data.startswith('STATE:'):
            with self.state_lock:
                self.mission_status = msg.data[6:]

    def lidar_callback(self, msg):
        # Sensor-frame projection, +/-10 m. Measured robot forward is down in this view.
        fields = point_cloud2.read_points(msg, field_names=('x', 'y', 'z')).reshape(-1)
        points = np.column_stack([fields[name] for name in ('x', 'y', 'z')])
        points = points[np.isfinite(points).all(axis=1) & (np.abs(points[:, :2]) < 10).all(axis=1)]
        pixels = (200 - points[:, [1, 0]] * 20).astype(int)
        # Fixed sensor-relative scale prevents colors shifting between frames.
        heights = np.full(401 * 401, -np.inf)
        np.maximum.at(heights, pixels[:, 1] * 401 + pixels[:, 0], points[:, 2])
        occupied = np.isfinite(heights)
        levels = np.zeros(heights.shape, dtype=np.uint8)
        levels[occupied] = (np.clip((heights[occupied] + 1.0) / 3.0, 0, 1) * 255).astype(np.uint8)
        # OpenCV outputs BGR; the LiDAR QImage below expects RGB.
        colors = cv2.applyColorMap(levels.reshape(401, 401), cv2.COLORMAP_JET)[:, :, ::-1]
        image = np.zeros((401, 401, 3), dtype=np.uint8)
        mask = occupied.reshape(401, 401)
        image[mask] = colors[mask]
        cv2.putText(image, 'Height: blue <= -1m / red >= +2m', (8, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # Center-mounted LiDAR: 0.52 m front/back, 0.49 m across, at 20 px/m.
        # Subpixel coordinates preserve the measured dimensions before rasterizing.
        half_width, half_length = 0.49 * 20 / 2, 0.52 * 20 / 2
        corner_min = (round((200-half_width)*256), round((200-half_length)*256))
        corner_max = (round((200+half_width)*256), round((200+half_length)*256))
        cv2.rectangle(image, corner_min, corner_max, (255, 255, 255), 1, shift=8)
        cv2.arrowedLine(image, (200, 198), (200, 204), (255, 255, 255), 1, tipLength=0.5)
        with self.state_lock:
            self.latest_lidar_image = image

    def telemetry_callback(self, msg):
        with self.state_lock:
            self.last_command_time = time.monotonic()
            self.linear_speed = msg.linear.x
            self.angular_speed = msg.angular.z
        
    def imu_callback(self, msg):
        with self.state_lock:
            self.imu = msg
            self.last_imu_time = time.monotonic()

    def send_command(self, cmd):
        msg = String()
        msg.data = cmd
        self.mission_pub.publish(msg)
        self.get_logger().info(f"Sent {cmd} to robot")


class BasestationGUI(QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("Robot Basestation Control")
        self.resize(1000, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column: Video
        self.video_label = QLabel("Waiting for Camera Feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.video_label, stretch=2)

        # Right Column
        right_layout = QVBoxLayout()
        
        # Telemetry
        telemetry_group = QGroupBox("Motion Telemetry")
        telemetry_inner = QVBoxLayout()
        self.linear_label = QLabel("Requested Linear (m/s): 0.00")
        self.requested_angular_label = QLabel("Requested Angular (rad/s): 0.00")
        self.angular_label = QLabel("IMU gyro: waiting")
        self.accel_label = QLabel("IMU acceleration: waiting")
        self.imu_frame_label = QLabel("IMU frame: waiting")
        for label in (self.linear_label, self.requested_angular_label,
                      self.angular_label, self.accel_label, self.imu_frame_label):
            label.setStyleSheet("font-size: 16px; font-weight: bold;")
            label.setWordWrap(True)
        telemetry_inner.addWidget(self.linear_label)
        telemetry_inner.addWidget(self.requested_angular_label)
        telemetry_inner.addWidget(self.angular_label)
        telemetry_inner.addWidget(self.accel_label)
        telemetry_inner.addWidget(self.imu_frame_label)
        telemetry_group.setLayout(telemetry_inner)
        right_layout.addWidget(telemetry_group)

        # Mission Controls
        control_group = QGroupBox("Auto Navigation")
        control_inner = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Auto Mission")
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; font-size: 16px; height: 50px;")
        self.start_btn.clicked.connect(lambda: self.node.send_command('START'))
        
        self.estop_btn = QPushButton("E-STOP")
        self.estop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; font-size: 20px; height: 60px;")
        self.estop_btn.clicked.connect(lambda: self.node.send_command('ESTOP'))
        
        self.manual_btn = QPushButton("Manual Mode / Reset Stop")
        self.manual_btn.clicked.connect(lambda: self.node.send_command('MANUAL'))
        self.status_label = QLabel("Waiting for robot")
        self.status_label.setWordWrap(True)
        control_inner.addWidget(self.status_label)
        control_inner.addWidget(self.manual_btn)
        control_inner.addWidget(self.start_btn)
        control_inner.addWidget(self.estop_btn)
        control_group.setLayout(control_inner)
        right_layout.addWidget(control_group)

        # LiDAR
        lidar_group = QGroupBox("VLP-16 height (sensor frame, ±10 m)")
        self.lidar_label = QLabel("Waiting for /reduced_lidar_points")
        self.lidar_label.setAlignment(Qt.AlignCenter)
        self.lidar_label.setStyleSheet("background-color: #2b2b2b; color: #00ff00;")
        lidar_layout = QVBoxLayout()
        lidar_layout.addWidget(self.lidar_label)
        lidar_group.setLayout(lidar_layout)
        right_layout.addWidget(lidar_group, stretch=1)

        main_layout.addLayout(right_layout, stretch=1)

        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.refresh_display)
        self.display_timer.start(30)

    def refresh_display(self):
        # Transfer references only; decoding, overlays and LiDAR run outside Qt.
        # Clearing slots under the lock cannot discard a newly published frame.
        with self.node.state_lock:
            camera_image = self.node.latest_cv_image
            image = self.node.latest_lidar_image
            self.node.latest_cv_image = None
            self.node.latest_lidar_image = None
            last_command_time = self.node.last_command_time
            mission_status = self.node.mission_status
            linear_speed = self.node.linear_speed
            angular_speed = self.node.angular_speed
            imu = self.node.imu
            last_imu_time = self.node.last_imu_time

        if camera_image is not None:
            rgb_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

        if image is not None:
            qt_image = QImage(image.data, 401, 401, image.strides[0], QImage.Format_RGB888)
            self.lidar_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.lidar_label.size(), Qt.KeepAspectRatio))
        connected = time.monotonic() - last_command_time < 1.0
        self.status_label.setText(mission_status if connected else 'Robot commands unavailable')
        if not connected:
            linear_speed = 0.0
            angular_speed = 0.0
        self.linear_label.setText(f"Requested Linear (m/s): {linear_speed:.2f}")
        self.requested_angular_label.setText(f"Requested Angular (rad/s): {angular_speed:.2f}")
        if imu is not None and time.monotonic() - last_imu_time < 1.0:
            gyro = imu.angular_velocity
            accel = imu.linear_acceleration
            self.angular_label.setText(f"Gyro x/y/z (rad/s): {gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f}")
            self.accel_label.setText(f"Accel x/y/z (m/s², incl. gravity): {accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}")
            self.imu_frame_label.setText(f"IMU frame: {imu.header.frame_id}")
        else:
            self.angular_label.setText("IMU gyro: unavailable")
            self.accel_label.setText("IMU acceleration: unavailable")
            self.imu_frame_label.setText("IMU frame: unavailable")


def main():
    rclpy.init()
    node = BasestationNode()
    app = QApplication(sys.argv)
    gui = BasestationGUI(node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, name='basestation-ros')
    ros_thread.start()
    try:
        gui.show()
        app.exec_()
    finally:
        gui.display_timer.stop()
        # Finish callbacks before destroying their node and publishers.
        executor.shutdown()
        ros_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
