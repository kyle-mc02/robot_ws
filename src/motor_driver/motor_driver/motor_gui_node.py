import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox)
from PyQt5.QtCore import QTimer, Qt

class MotorControlUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Tank Drive Motor Controller")
        self.resize(350, 450)

        # Pololu VNH5019 library uses -400 to 400 for speed
        self.max_speed = 400 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        sliders_layout = QHBoxLayout()

        # 1. Left Motor Slider
        left_group = QGroupBox("Left Wheel")
        left_layout = QVBoxLayout()
        
        self.left_label = QLabel("0")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.left_slider = QSlider(Qt.Vertical)
        self.left_slider.setRange(-self.max_speed, self.max_speed)
        self.left_slider.setValue(0)
        self.left_slider.setTickPosition(QSlider.TicksBothSides)
        self.left_slider.setTickInterval(100)
        
        left_layout.addWidget(self.left_label)
        left_layout.addWidget(self.left_slider, alignment=Qt.AlignHCenter)
        left_group.setLayout(left_layout)
        sliders_layout.addWidget(left_group)

        # 2. Right Motor Slider
        right_group = QGroupBox("Right Wheel")
        right_layout = QVBoxLayout()
        
        self.right_label = QLabel("0")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.right_slider = QSlider(Qt.Vertical)
        self.right_slider.setRange(-self.max_speed, self.max_speed)
        self.right_slider.setValue(0)
        self.right_slider.setTickPosition(QSlider.TicksBothSides)
        self.right_slider.setTickInterval(100)
        
        right_layout.addWidget(self.right_label)
        right_layout.addWidget(self.right_slider, alignment=Qt.AlignHCenter)
        right_group.setLayout(right_layout)
        sliders_layout.addWidget(right_group)

        main_layout.addLayout(sliders_layout)

        # 3. Emergency Stop / Zero Motors Button
        self.stop_btn = QPushButton("ZERO MOTORS")
        self.stop_btn.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold; padding: 15px; font-size: 14px;")
        self.stop_btn.clicked.connect(self.stop_motors)
        main_layout.addWidget(self.stop_btn)

        # Signal connections
        self.left_slider.valueChanged.connect(self.update_motors)
        self.right_slider.valueChanged.connect(self.update_motors)

        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros_node)
        self.ros_timer.start(10)

    def spin_ros_node(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0)

    def stop_motors(self):
        self.left_slider.blockSignals(True)
        self.right_slider.blockSignals(True)
        
        self.left_slider.setValue(0)
        self.right_slider.setValue(0)
        
        self.left_slider.blockSignals(False)
        self.right_slider.blockSignals(False)
        
        self.update_motors()

    def update_motors(self):
        left_speed = self.left_slider.value()
        right_speed = self.right_slider.value()

        self.left_label.setText(f"{left_speed}")
        self.right_label.setText(f"{right_speed}")

        self.ros_node.publish_cmds(float(left_speed), float(right_speed))

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_gui_node')
        # Reusing Vector3 format: x = left motor, y = right motor
        self.cmd_pub = self.create_publisher(Vector3, '/motor_cmds', 10)

    def publish_cmds(self, left, right):
        msg = Vector3()
        msg.x = left
        msg.y = right
        msg.z = 0.0
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ros_node = MotorControlNode()
    app = QApplication(sys.argv)
    window = MotorControlUI(ros_node)
    window.show()
    exit_code = app.exec_()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
