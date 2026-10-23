#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
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
        self.linear_speed = 0.0
        self.angular_speed = 0.0

        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.image_callback, video_qos)
        self.create_subscription(Twist, '/cmd_vel', self.telemetry_callback, 10)
        
        # Publisher for Mission Control
        self.mission_pub = self.create_publisher(String, '/mission_control', 10)

    def image_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass

    def telemetry_callback(self, msg):
        self.linear_speed = msg.linear.x
        self.angular_speed = msg.angular.z
        
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
        telemetry_group = QGroupBox("Live Telemetry")
        telemetry_inner = QVBoxLayout()
        self.linear_label = QLabel("Linear Speed (m/s): 0.00")
        self.angular_label = QLabel("Angular Speed (rad/s): 0.00")
        self.linear_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.angular_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        telemetry_inner.addWidget(self.linear_label)
        telemetry_inner.addWidget(self.angular_label)
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
        
        control_inner.addWidget(self.start_btn)
        control_inner.addWidget(self.estop_btn)
        control_group.setLayout(control_inner)
        right_layout.addWidget(control_group)

        # LiDAR
        lidar_group = QGroupBox("VLP-16 LiDAR (Top-Down)")
        self.lidar_label = QLabel("LiDAR Visualizer\n(Pending Implementation)")
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
        rclpy.spin_once(self.node, timeout_sec=0.0)
        
        if self.node.latest_cv_image is not None:
            rgb_image = cv2.cvtColor(self.node.latest_cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))
            self.node.latest_cv_image = None
            
        self.linear_label.setText(f"Linear Speed (m/s): {self.node.linear_speed:.2f}")
        self.angular_label.setText(f"Angular Speed (rad/s): {self.node.angular_speed:.2f}")


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
