#!/usr/bin/env python3

import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class JoyToCmdVel(Node):
    def __init__(self):
        super().__init__('joy_to_cmd_vel')

        # Keep these nominal values aligned with driver_node.py.
        self.declare_parameter('wheel_radius', 0.075)       # meters
        self.declare_parameter('wheel_separation', 0.44)   # meters
        self.declare_parameter('gear_ratio', 504.0)         # motor / wheel speed
        self.declare_parameter('max_motor_speed', 680.0 / 3 * math.pi)   # rad/s BEFORE gearbox

        # Axis indexes depend on the joystick device; verify against /joy.
        self.declare_parameter('forward_axis', 1)  # Left stick vertical
        self.declare_parameter('turn_axis', 2)     # Right stick horizontal

        # Set either sign to -1.0 if its direction is reversed.
        self.declare_parameter('forward_sign', 1.0)
        self.declare_parameter('turn_sign', 1.0)

        self.radius = self.get_parameter('wheel_radius').value
        self.separation = self.get_parameter('wheel_separation').value
        self.gear_ratio = self.get_parameter('gear_ratio').value
        self.max_motor_speed = self.get_parameter('max_motor_speed').value

        if min(
            self.radius,
            self.separation,
            self.gear_ratio,
            self.max_motor_speed,
        ) <= 0:
            raise ValueError('Robot dimensions and speed limit must be positive.')

        self.forward_axis = self.get_parameter('forward_axis').value
        self.turn_axis = self.get_parameter('turn_axis').value
        self.forward_sign = self.get_parameter('forward_sign').value
        self.turn_sign = self.get_parameter('turn_sign').value

        self.deadzone = 0.1
        self.timeout = 0.5
        self.last_received = None
        self.command = Twist()

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode = 'IDLE'
        self.mode_received = float('-inf')
        self.create_subscription(String, '/mission_control', self.mode_callback, 10)

        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(0.05, self.publish_command)

    def mode_callback(self, msg):
        if not msg.data.startswith('STATE:'):
            return
        mode = msg.data[6:]
        if mode != self.mode:
            self.last_received = None  # Require fresh joystick input after a mode change.
        self.mode = mode
        self.mode_received = time.monotonic()

    def apply_deadzone(self, value):
        return 0.0 if abs(value) < self.deadzone else value

    def joy_callback(self, msg):
        if not (
            0 <= self.forward_axis < len(msg.axes)
            and 0 <= self.turn_axis < len(msg.axes)
        ):
            self.command = Twist()
            self.last_received = None
            return

        forward = self.forward_sign * self.apply_deadzone(
            msg.axes[self.forward_axis]
        )
        turn = self.turn_sign * self.apply_deadzone(
            msg.axes[self.turn_axis]
        )

        # Mix forward and turning inputs into left/right motor requests.
        # Positive turn: right wheel faster, so robot turns left.
        left = forward - turn
        right = forward + turn

        # Keep both requests within [-1, 1], preserving their ratio.
        scale = max(1.0, abs(left), abs(right))
        left /= scale
        right /= scale

        # Requested motor angular velocities, in rad/s.
        omega_left = left * self.max_motor_speed
        omega_right = right * self.max_motor_speed

        velocity = (
            self.radius / (2.0 * self.gear_ratio)
        ) * (omega_left + omega_right)

        angular_velocity = (
            self.radius / (self.separation * self.gear_ratio)
        ) * (omega_right - omega_left)

        self.command = Twist()
        self.command.linear.x = velocity
        self.command.angular.z = angular_velocity
        self.last_received = time.monotonic()

    def publish_command(self):
        # Stay silent while autonomy owns /cmd_vel, or if its status is lost.
        if self.mode != 'MANUAL' or time.monotonic() - self.mode_received > self.timeout:
            return
        if (
            self.last_received is None
            or time.monotonic() - self.last_received > self.timeout
        ):
            self.publisher.publish(Twist())
        else:
            self.publisher.publish(self.command)


def main(args=None):
    rclpy.init(args=args)
    node = JoyToCmdVel()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok() and node.mode == 'MANUAL':
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
