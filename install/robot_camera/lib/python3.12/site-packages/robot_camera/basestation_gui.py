import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class BasestationGUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Robot Base Station Camera Monitor")
        self.resize(700, 550)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Camera Feed Display UI
        self.image_label = QLabel("Waiting for camera feed...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: #222; color: white;")
        main_layout.addWidget(self.image_label, stretch=1)

        # Thread-safe ROS 2 Spinning via PyQt Timer
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros_node)
        self.ros_timer.start(10) # 100Hz loop

    def spin_ros_node(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0) #

    def update_image(self, cv_image):
        # Convert BGR (OpenCV standard) to RGB (Qt standard)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888) #
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)) #

class ROS2BasestationNode(Node):
    def __init__(self):
        super().__init__('basestation_gui_node')
        self.ui = None 

        # Subscribe directly to the compressed stream
        self.create_subscription(
            CompressedImage, 
            '/camera/image_compressed', 
            self.image_callback, 
            10
        )

    def image_callback(self, msg):
        if self.ui:
            # Decode JPEG memory buffer directly back into an OpenCV matrix image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.ui.update_image(cv_image)

def main(args=None):
    rclpy.init(args=args)
    ros_node = ROS2BasestationNode()
    
    app = QApplication(sys.argv)
    window = BasestationGUI(ros_node)
    ros_node.ui = window 
    
    window.show()
    exit_code = app.exec_()
    
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()

