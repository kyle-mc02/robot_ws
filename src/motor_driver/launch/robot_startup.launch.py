import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Velodyne VLP-16 Launch setup
    velodyne_share_dir = get_package_share_directory('velodyne')
    velodyne_launch_file = os.path.join(velodyne_share_dir, 'launch', 'velodyne-all-nodes-VLP16-launch.py')

    # 2. RealSense D435i Launch setup
    realsense_share_dir = get_package_share_directory('realsense2_camera')
    realsense_launch_file = os.path.join(realsense_share_dir, 'launch', 'rs_launch.py')

    return LaunchDescription([
        # Motor Driver Node
        Node(
            package='motor_driver',
            executable='driver_node',
            name='robot_driver',
            output='screen'
        ),
        
        # Auto Navigation Listener
        Node(
            package='motor_driver',
            executable='auto_nav_node',
            name='auto_nav_node',
            output='screen'
        ),

        # RealSense D435i Camera
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            # Optional: Pass launch arguments to enable the built-in IMU on the D435i
            launch_arguments={
                'enable_gyro': 'true', 
                'enable_accel': 'true',
                'align_depth.enable': 'true',
                'rgb_camera.color_profile': '640x480x15',
                'depth_module.depth_profile': '640x480x15'
            }.items()
        ),

        # Velodyne VLP-16 LiDAR
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(velodyne_launch_file)
        )
    ])

