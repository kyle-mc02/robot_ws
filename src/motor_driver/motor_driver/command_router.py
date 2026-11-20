"""Select manual or Nav2 velocity commands; start stopped."""
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class CommandRouter(Node):
    def __init__(self):
        super().__init__('command_router')
        self.mode = 'ESTOP'
        self.command = Twist()
        self.received = float('-inf')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status = self.create_publisher(String, '/mission_control', 10)
        self.create_subscription(String, '/mission_control', self.select, 10)
        self.create_subscription(Twist, '/cmd_vel_teleop', lambda m: self.accept(m, 'MANUAL'), 10)
        self.create_subscription(Twist, '/navigation/cmd_vel', lambda m: self.accept(m, 'NAV2'), 10)
        self.create_timer(.05, self.tick)

    def select(self, msg):
        if msg.data in ('MANUAL', 'NAV2', 'ESTOP'):
            self.mode = msg.data
            self.command = Twist()
            self.received = float('-inf')
            self.pub.publish(Twist())

    def accept(self, msg, mode):
        if self.mode == mode:
            self.command = msg
            self.received = time.monotonic()

    def tick(self):
        fresh = time.monotonic() - self.received < .3
        self.pub.publish(self.command if fresh and self.mode != 'ESTOP' else Twist())
        self.status.publish(String(data='STATE:' + self.mode))

def main(args=None):
    rclpy.init(args=args)
    node = CommandRouter()
    try:
        rclpy.spin(node)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
