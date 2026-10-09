import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import Vector3
import pyrealsense2 as rs
import cv2
import numpy as np

class RobotD435iPublisher(Node):
    def __init__(self):
        super().__init__('robot_d435i_publisher')
        
        # High-efficiency Wi-Fi publishers
        self.color_pub = self.create_publisher(CompressedImage, '/camera/color/compressed', 10)
        self.depth_pub = self.create_publisher(CompressedImage, '/camera/depth/compressed', 10)
        self.infra1_pub = self.create_publisher(CompressedImage, '/camera/infra1/compressed', 10)
        self.infra2_pub = self.create_publisher(CompressedImage, '/camera/infra2/compressed', 10)
        self.imu_pub = self.create_publisher(Imu, '/camera/imu/data', 10)

        # Initialize RealSense Pipeline & Config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 1. Enable Visual Video Streams (Optimized 640x480 resolution for network safety)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
        self.config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)

        # 2. Enable Motion Tracking IMU Streams
        self.config.enable_stream(rs.stream.gyro)
        self.config.enable_stream(rs.stream.accel)

        # Start the camera pipeline
        self.profile = self.pipeline.start(self.config)
        self.get_logger().info("Intel RealSense D435i pipeline fully active with all streams.")

        # Non-blocking callback timer (approx 30Hz loop to catch hardware frames)
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        # Poll for hardware framesets non-blockingly
        frames = self.pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        infra1_frame = frames.get_infrared_frame(1)
        infra2_frame = frames.get_infrared_frame(2)
        
        # Look for IMU motion frames in the bundle
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        # --- 1. Process Video Streams ---
        if color_frame:
            color_img = np.asanyarray(color_frame.get_data())
            msg = self.compress_image(color_img, 'jpeg')
            self.color_pub.publish(msg)

        if depth_frame:
            # Depth maps must be converted to an 8-bit visual scale or colormap to compress cleanly as JPEG
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_scaled = cv2.convertScaleAbs(depth_img, alpha=0.03) 
            msg = self.compress_image(depth_scaled, 'jpeg')
            self.depth_pub.publish(msg)

        if infra1_frame:
            infra1_img = np.asanyarray(infra1_frame.get_data())
            msg = self.compress_image(infra1_img, 'jpeg')
            self.infra1_pub.publish(msg)

        if infra2_frame:
            infra2_img = np.asanyarray(infra2_frame.get_data())
            msg = self.compress_image(infra2_img, 'jpeg')
            self.infra2_pub.publish(msg)

        # --- 2. Process & Merge IMU Streams ---
        if accel_frame and gyro_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = "camera_imu_optical_frame"

            # Angular velocities (rad/s)
            imu_msg.angular_velocity = Vector3(x=gyro_data.x, y=gyro_data.y, z=gyro_data.z)
            # Linear accelerations (m/s^2)
            imu_msg.linear_acceleration = Vector3(x=accel_data.x, y=accel_data.y, z=accel_data.z)

            self.imu_pub.publish(imu_msg)

    def compress_image(self, cv_matrix, format_str='jpeg'):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = format_str
        success, encoded_img = cv2.imencode(f'.{format_str}', cv_matrix)
        if success:
            msg.data = encoded_img.tobytes()
        return msg

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobotD435iPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

