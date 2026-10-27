import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct
import math
import time

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
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

