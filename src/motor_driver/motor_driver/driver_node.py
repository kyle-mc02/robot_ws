import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct
import math

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
            self.get_logger().info("Connected to Robot Arduino")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.serial_port = None

        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.listener_callback, 10)
    
    # Expecting a twist with (Linear vel, Angular vel)
    def listener_callback(self, msg):
        # Motor = 13.5 RPM = 27/2 * 2pi/60 : pi*9/20 
        # Rated RMP = 6800 (max is 7000) = 6800 * 2pi/60 : rad/s (680/3*pi)
        MAX_RAD_S = 680/3 * math.pi # in rad/s
        MAX_PWM = 340.0 # True max is 400
        
        eta = 504 # Gear ratio
        Rwheel = 0.15/2 #in m (radius of wheels)
        lwheels = 0.44  #in m (distance between wheels)
        # linFactor, angFactor = 1.0, 1.0 # Scaling factors to tune velocitites.
        
        linVel, angVel  = float(msg.linear.x), float(msg.angular.z) 

        wml = (eta/Rwheel) * (linVel - 0.5*lwheels*angVel)
        wmr = (eta/Rwheel) * (linVel + 0.5*lwheels*angVel)
        
        # Constrain values to the shield's limits to prevent integer overflow
        left = max(-MAX_PWM, min(MAX_PWM, int((wml / MAX_RAD_S) * MAX_PWM)))
        right = max(-MAX_PWM, min(MAX_PWM, int((-wmr / MAX_RAD_S) * MAX_PWM)))
        
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

