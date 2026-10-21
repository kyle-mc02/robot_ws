#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, QThread, Qt
import cv2
import numpy as np

# --- 1. ROS 2 Background Thread ---
class ROS2Thread(QThread, Node):
    # Signals to send data safely to the GUI thread
    image_signal = pyqtSignal(np.ndarray)
    telemetry_signal = pyqtSignal(float, float)

    def __init__(self):
        QThread.__init__(self)
        Node.__init__(self, 'basestation_gui_node')
        self.bridge = CvBridge()

        # Subscribe to the camera (Adjust topic name if yours is different)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        
        # Subscribe to cmd_vel to read out the current speed commands
        self.create_subscription(Twist, '/cmd_vel', self.telemetry_callback, 10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_signal.emit(cv_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def telemetry_callback(self, msg):
        self.telemetry_signal.emit(msg.linear.x, msg.angular.z)

    def run(self):
        rclpy.spin(self)


# --- 2. PyQt5 Main Window ---
class BasestationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Basestation Control")
        self.resize(1000, 600)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column: Video Feed
        self.video_label = QLabel("Waiting for Camera Feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        main_layout.addWidget(self.video_label, stretch=2)

        # Right Column: Telemetry & LiDAR Placeholder
        right_layout = QVBoxLayout()
        
        # Telemetry Box
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

        # LiDAR Placeholder Box
        lidar_group = QGroupBox("VLP-16 LiDAR (Top-Down)")
        self.lidar_label = QLabel("LiDAR Visualizer\n(Pending Implementation)")
        self.lidar_label.setAlignment(Qt.AlignCenter)
        self.lidar_label.setStyleSheet("background-color: #2b2b2b; color: #00ff00;")
        lidar_layout = QVBoxLayout()
        lidar_layout.addWidget(self.lidar_label)
        lidar_group.setLayout(lidar_layout)
        right_layout.addWidget(lidar_group, stretch=1)

        main_layout.addLayout(right_layout, stretch=1)

        # --- 3. Start ROS 2 Thread ---
        self.ros_thread = ROS2Thread()
        self.ros_thread.image_signal.connect(self.update_image)
        self.ros_thread.telemetry_signal.connect(self.update_telemetry)
        self.ros_thread.start()

    def update_image(self, cv_image):
        # Convert OpenCV BGR to Qt RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit the label without distorting aspect ratio
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def update_telemetry(self, linear, angular):
        self.linear_label.setText(f"Linear Speed (m/s): {linear:.2f}")
        self.angular_label.setText(f"Angular Speed (rad/s): {angular:.2f}")

    def closeEvent(self, event):
        # Clean up ROS 2 on window close
        self.ros_thread.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    gui = BasestationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
