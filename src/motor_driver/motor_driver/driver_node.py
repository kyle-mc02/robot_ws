import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct

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
            Twist, '/motor_cmds', self.listener_callback, 10)
    
    # Expecting a vector with (Linear vel, Angular val, ___)
    def listener_callback(self, msg):
        # Motor = 13.5 RPM = 27/2 * 2pi/60 : pi*9/20 
        # Rated RMP = 6800 (max is 7000) = 6800 * 2pi/60 : rad/s (680/3*pi)
        linVel = float(msg.linear.x)
        angVel = float(msg.angular.z) 

        eta = 504
        Rwheel = 0.15/2 #in m
        lwheels = 0.44  #in m
        MAX = 6800
        
        wml = (eta/Rwheel) * (linVel - 0.5*lwheels*angVel)
        wmr = (eta/Rwheel) * (linVel + 0.5*lwheels*angVel)
        
        # Constrain values to the shield's limits to prevent integer overflow
        left = max(-340, min(340, int(wml / MAX)))
        right = max(-340, min(340, int(-wmr / MAX)))
        
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

