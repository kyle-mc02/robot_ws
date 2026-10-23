from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Hardware Joystick Driver
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'deadzone': 0.05,
                'autorepeat_rate': 20.0,
            }],
            output='screen'
        ),

        # 2. Basestation PyQt5 Visualizer
        Node(
            package='basestation_visualization',
            executable='gui_node',
            name='gui_node',
            output='screen'
        )
    ])
