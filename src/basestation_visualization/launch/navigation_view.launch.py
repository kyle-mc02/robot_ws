"""Nav2 RViz on the base station."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = Path(get_package_share_directory('basestation_visualization')) / 'config/navigation.rviz'
    return LaunchDescription([Node(package='rviz2', executable='rviz2', arguments=['-d', str(config)], output='screen')])
