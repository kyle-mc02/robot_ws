import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct
import math
import time
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        self.declare_parameter('lidar_max_points', 2000)
        self.add_on_set_parameters_callback(self.validate_lidar_parameters)
        initial_limit = self.get_parameter('lidar_max_points').value
        if type(initial_limit) is not int or not 1 <= initial_limit <= 100000:
            raise ValueError('lidar_max_points must be an integer from 1 to 100000')
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1, write_timeout=0.1)
            self.get_logger().info("Connected to Robot Arduino")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            raise RuntimeError('Motor driver cannot run without its serial connection.') from e

        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.listener_callback, 10)

        self.last_received = time.monotonic()
        self.create_timer(0.1, self.check_timeout)

        # All rings are reduced locally; only XYZ crosses Wi-Fi to the GUI.
        self.latest_cloud = None
        self.lidar_pub = self.create_publisher(
            PointCloud2, '/reduced_lidar_points', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/velodyne_points',
                                 self.lidar_callback, qos_profile_sensor_data)
        self.create_timer(0.2, self.publish_reduced_lidar)  # Maximum 5 Hz.

    def validate_lidar_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == 'lidar_max_points':
                value = parameter.value
                if type(value) is not int or not 1 <= value <= 100000:
                    return SetParametersResult(successful=False,
                                               reason='lidar_max_points must be an integer from 1 to 100000')
        return SetParametersResult(successful=True)

    def lidar_callback(self, msg):
        self.latest_cloud = msg  # Never queue old clouds for conversion.

    def publish_reduced_lidar(self):
        if self.latest_cloud is None:
            return
        cloud, self.latest_cloud = self.latest_cloud, None
        try:
            fields = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z')).reshape(-1)
            points = np.column_stack([fields[name] for name in ('x', 'y', 'z')])
            # Keep the GUI's +/-10 m sensor-frame view, using every ring.
            points = points[np.isfinite(points).all(axis=1)
                            & (np.abs(points[:, :2]) < 10).all(axis=1)]
            # One representative per 5 cm XY cell, matching the existing display.
            cells = np.floor(points[:, :2] / 0.05).astype(np.int32)
            cell_ids = (cells[:, 0] + 200) * 400 + cells[:, 1] + 200
            _, indices = np.unique(cell_ids, return_index=True)
            points = points[indices]
            limit = self.get_parameter('lidar_max_points').value
            if len(points) > limit:
                points = points[np.linspace(0, len(points)-1, limit, dtype=int)]
            self.lidar_pub.publish(point_cloud2.create_cloud_xyz32(
                cloud.header, np.asarray(points, dtype=np.float32)))
        except (ValueError, AssertionError) as error:
            self.get_logger().warning(f'LiDAR reduction failed: {error}')

    def check_timeout(self):
        if time.monotonic() - self.last_received > 0.5:
            self.listener_callback(Twist())

    # Expecting a twist with (Linear vel, Angular vel)
    def listener_callback(self, msg):
        # Keep these nominal values aligned with controller_node.py.
        eta, Rwheel, lwheels = 504.0, 0.075, 0.44
        MAX_MOTOR_SPEED = 680.0 / 3 * math.pi
        MAX_PWM = 340.0
        self.last_received = time.monotonic()

        linVel, angVel  = float(msg.linear.x), float(msg.angular.z) 

        if not math.isfinite(linVel) or not math.isfinite(angVel):
            linVel = angVel = 0.0

        wml = (eta/Rwheel) * (linVel - 0.5*lwheels*angVel)
        wmr = (eta/Rwheel) * (linVel + 0.5*lwheels*angVel)
        
        # Scale both sides together so saturation preserves the requested curvature.
        scale = max(1.0, abs(wml) / MAX_MOTOR_SPEED, abs(wmr) / MAX_MOTOR_SPEED)
        wml /= scale
        wmr /= scale
        # Right motor is wired in the opposite direction.
        left = int(max(-MAX_PWM, min(MAX_PWM, (wml / MAX_MOTOR_SPEED) * MAX_PWM)))
        right = int(max(-MAX_PWM, min(MAX_PWM, (-wmr / MAX_MOTOR_SPEED) * MAX_PWM)))
        
        if self.serial_port and self.serial_port.is_open:
            # Pack format: '<Bhh' 
            # '<' = Little-endian (Standard for Arduino)
            # 'B' = Unsigned char (1 byte) -> We use 0xAA as a start marker
            # 'h' = Signed short (2 bytes) -> Left motor
            # 'h' = Signed short (2 bytes) -> Right motor
            packet = struct.pack('<Bhh', 0xAA, left, right)
            
            self.serial_port.write(packet)
            self.serial_port.flush()

def main(args=None):
    rclpy.init(args=args)
    node = RobotDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial_port and node.serial_port.is_open:
            # Send binary stop command (Start byte + 0 + 0)
            node.serial_port.write(struct.pack('<Bhh', 0xAA, 0, 0))
            node.serial_port.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

