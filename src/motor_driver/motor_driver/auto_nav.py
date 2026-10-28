#!/usr/bin/env python3
"""Run the hardcoded timed mission; leave /cmd_vel to the joystick in manual mode."""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class AutoNavNode(Node):
    def __init__(self):
        super().__init__('auto_nav_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mission_pub = self.create_publisher(String, '/mission_control', 10)
        self.create_subscription(String, '/mission_control', self.control_callback, 10)
        self.mode = 'ESTOP'
        self.steps = []
        self.deadline = 0.0
        self.command = Twist()
        self.create_timer(0.05, self.publish_command)

    def control_callback(self, msg):
        if msg.data not in ('START', 'ESTOP', 'MANUAL'):
            return  # Ignore our own STATE messages.
        if msg.data == 'START' and self.mode != 'AUTO':
            self.steps = self.build_course()
            self.mode = 'AUTO'
            self.deadline = 0.0
            self.command = Twist()
        elif msg.data in ('ESTOP', 'MANUAL'):
            self.mode = msg.data
            self.steps = []
            self.cmd_pub.publish(Twist())
        self.publish_command()

    def build_course(self):
        # (linear request, angular request, seconds). Speeds are not measured.
        course = [
            (0.05, 0.0, 6.8),  # Tuned 3 m forward.
            (0.0, 0.2, 2.0),   # Tuned left 90 degrees.
            (0.05, 0.0, 6.75 * 7.0 / 3.0),  # Estimated 7 m; validate on floor.
        ]
        steps = []
        for linear, angular, duration in course:
            command = Twist()
            command.linear.x, command.angular.z = linear, angular
            steps.extend([(command, duration), (Twist(), 0.5)])
        return steps

    def publish_command(self):
        now = time.monotonic()
        command = Twist()
        if self.mode == 'AUTO':
            if now >= self.deadline:
                if self.steps:
                    self.command, duration = self.steps.pop(0)
                    self.deadline = now + duration
                    self.get_logger().info(
                        f'AUTO segment: linear={self.command.linear.x:.3f} m/s, '
                        f'angular={self.command.angular.z:.3f} rad/s, duration={duration:.2f} s')
                else:
                    # Hold stopped until the operator explicitly selects a mode.
                    self.mode = 'IDLE'
                    self.get_logger().info('AUTO complete; holding stopped in IDLE')
                    self.command = Twist()
            command = self.command
        self.mission_pub.publish(String(data='STATE:' + self.mode))
        if self.mode != 'MANUAL':
            self.cmd_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = AutoNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
