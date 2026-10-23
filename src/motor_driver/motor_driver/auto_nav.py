#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import threading
import math

class AutoNavNode(Node):
    def __init__(self):
        super().__init__('auto_nav_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_sub = self.create_subscription(String, '/mission_control', self.control_callback, 10)
        
        self.mission_thread = None
        self.estop_active = False
        self.mission_running = False

        self.get_logger().info("Auto Nav Node ready. Waiting for START command from GUI...")

    def control_callback(self, msg):
        if msg.data == 'START':
            if not self.mission_running:
                self.get_logger().info("Mission START received!")
                self.estop_active = False
                self.mission_running = True
                self.mission_thread = threading.Thread(target=self.execute_course)
                self.mission_thread.start()
            else:
                self.get_logger().warn("Mission is already running!")
                
        elif msg.data == 'ESTOP':
            self.get_logger().error("E-STOP TRIGGERED! HALTING IMMEDIATELY.")
            self.estop_active = True
            self.stop_robot()

    def execute_course(self):
        # ---------------------------------------------------------
        # TUNING SEQUENCE
        # ---------------------------------------------------------
        course = [
            {'type': 'drive', 'amount': 1.0,  'speed': 0.3},
            {'type': 'turn',  'amount': 90.0, 'speed': 0.5},
            {'type': 'drive', 'amount': 1.0,  'speed': 0.3},
            {'type': 'turn',  'amount': -90.0,'speed': 0.5},
            {'type': 'drive', 'amount': 1.0,  'speed': 0.3}
        ]

        for step in course:
            if self.estop_active or not rclpy.ok():
                self.get_logger().warn("Mission aborted mid-course.")
                break
            self.perform_step(step)

        self.stop_robot()
        self.mission_running = False
        if not self.estop_active:
            self.get_logger().info("Course completed successfully!")

    def perform_step(self, step):
        msg = Twist()
        
        if step['type'] == 'drive':
            duration = abs(step['amount'] / step['speed'])
            msg.linear.x = step['speed'] if step['amount'] > 0 else -step['speed']
            self.get_logger().info(f"Driving {step['amount']}m at {step['speed']}m/s")
            
        elif step['type'] == 'turn':
            radians = math.radians(step['amount'])
            duration = abs(radians / step['speed'])
            msg.angular.z = step['speed'] if step['amount'] > 0 else -step['speed']
            self.get_logger().info(f"Turning {step['amount']} deg at {step['speed']}rad/s")

        # Interruptible loop: Check for E-Stop 20 times a second
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.estop_active or not rclpy.ok():
                return  # Bail out instantly
            
            # Continuously publish to override any joystick noise
            self.cmd_pub.publish(msg)
            time.sleep(0.05) 
        
        self.stop_robot()
        if not self.estop_active:
            time.sleep(0.5)

    def stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
