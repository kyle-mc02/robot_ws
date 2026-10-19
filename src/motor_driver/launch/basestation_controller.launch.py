from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Official ROS 2 hardware driver (Spawns /joy)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0, # Sends /joy updates at 20Hz even if sticks are held still
            }],
            output='screen'
        ),
        
        # 2. Your team's custom translation script (Reads /joy, Spawns /cmd_vel)
        Node(
            package='motor_driver',   
            executable='controller_node', 
            name='controller_node',
            output='screen'
        )
    ])
