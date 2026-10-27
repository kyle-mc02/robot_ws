#!/usr/bin/env python3

import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2

class BasestationNode(Node):
    def __init__(self):
        super().__init__('basestation_gui_node')
        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.latest_lidar_image = None
        self.mission_status = 'Waiting for robot'
        self.last_command_time = float('-inf')
        self.linear_speed = 0.0
        self.imu = None
        self.last_imu_time = float('-inf')

        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.image_callback, video_qos)
        self.create_subscription(Twist, '/cmd_vel', self.telemetry_callback, 10)
        self.create_subscription(String, '/mission_control', self.status_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, video_qos)
        self.create_subscription(Imu, '/camera/camera/imu', self.imu_callback, video_qos)
        
        # Publisher for Mission Control
        self.mission_pub = self.create_publisher(String, '/mission_control', 10)

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"Camera decode failed: {e}")

    def status_callback(self, msg):
        if msg.data.startswith('STATE:'):
            self.mission_status = msg.data[6:]

    def lidar_callback(self, msg):
        # Sensor coordinates: forward is up, left is left. Show +/-10 metres.
        ranges = np.asarray(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        ranges, angles = ranges[valid], angles[valid]
        points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        points = points[np.isfinite(points).all(axis=1) & (np.abs(points) < 10).all(axis=1)]
        pixels = (200 - points[:, ::-1] * 20).astype(int)
        image = np.zeros((401, 401, 3), dtype=np.uint8)
        image[pixels[:, 1], pixels[:, 0]] = (0, 255, 0)
        cv2.circle(image, (200, 200), 4, (255, 255, 255), -1)
        self.latest_lidar_image = image

    def telemetry_callback(self, msg):
        self.last_command_time = time.monotonic()
        self.linear_speed = msg.linear.x
        
    def imu_callback(self, msg):
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
        self.angular_label = QLabel("IMU gyro: waiting")
        self.accel_label = QLabel("IMU acceleration: waiting")
        self.imu_frame_label = QLabel("IMU frame: waiting")
        self.linear_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.angular_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        telemetry_inner.addWidget(self.linear_label)
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
        lidar_group = QGroupBox("VLP-16 scan (one ring, ±10 m)")
        self.lidar_label = QLabel("Waiting for /scan")
        self.lidar_label.setAlignment(Qt.AlignCenter)
        self.lidar_label.setStyleSheet("background-color: #2b2b2b; color: #00ff00;")
        lidar_layout = QVBoxLayout()
        lidar_layout.addWidget(self.lidar_label)
        lidar_group.setLayout(lidar_layout)
        right_layout.addWidget(lidar_group, stretch=1)

        main_layout.addLayout(right_layout, stretch=1)

        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(30)

    def spin_ros(self):
        # Drain enough callbacks for camera, commands, status and LiDAR.
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.0)
        
        if self.node.latest_cv_image is not None:
            rgb_image = cv2.cvtColor(self.node.latest_cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))
            self.node.latest_cv_image = None
            
        if self.node.latest_lidar_image is not None:
            image = self.node.latest_lidar_image
            qt_image = QImage(image.data, 401, 401, image.strides[0], QImage.Format_RGB888)
            self.lidar_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.lidar_label.size(), Qt.KeepAspectRatio))
            self.node.latest_lidar_image = None
        connected = time.monotonic() - self.node.last_command_time < 1.0
        self.status_label.setText(self.node.mission_status if connected else 'Robot commands unavailable')
        if not connected:
            self.node.linear_speed = 0.0
        self.linear_label.setText(f"Requested Linear (m/s): {self.node.linear_speed:.2f}")
        if time.monotonic() - self.node.last_imu_time < 1.0:
            gyro = self.node.imu.angular_velocity
            accel = self.node.imu.linear_acceleration
            self.angular_label.setText(f"Gyro x/y/z (rad/s): {gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f}")
            self.accel_label.setText(f"Accel x/y/z (m/s², incl. gravity): {accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}")
            self.imu_frame_label.setText(f"IMU frame: {self.node.imu.header.frame_id}")
        else:
            self.angular_label.setText("IMU gyro: unavailable")
            self.accel_label.setText("IMU acceleration: unavailable")
            self.imu_frame_label.setText("IMU frame: unavailable")


def main():
    rclpy.init()
    node = BasestationNode()
    app = QApplication(sys.argv)
    gui = BasestationGUI(node)
    gui.show()
    app.exec_()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
