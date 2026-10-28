import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Start only Velodyne acquisition and point-cloud conversion, not LaserScan.
    driver_share = get_package_share_directory('velodyne_driver')
    cloud_share = get_package_share_directory('velodyne_pointcloud')
    driver_config = os.path.join(driver_share, 'config', 'VLP16-velodyne_driver_node-params.yaml')
    cloud_config = os.path.join(cloud_share, 'config', 'VLP16-velodyne_transform_node-params.yaml')
    with open(cloud_config) as config:
        cloud_parameters = yaml.safe_load(config)['velodyne_transform_node']['ros__parameters']
    cloud_parameters['calibration'] = os.path.join(cloud_share, 'params', 'VLP16db.yaml')
    velodyne_driver = Node(package='velodyne_driver', executable='velodyne_driver_node',
                           output='screen', parameters=[driver_config])
    velodyne_cloud = Node(package='velodyne_pointcloud', executable='velodyne_transform_node',
                          output='screen', parameters=[cloud_parameters])

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
            executable='auto_nav',
            name='auto_nav',
            output='screen'
        ),

        Node(
            package='motor_driver',   
            executable='controller_node', 
            name='controller_node',
            output='screen'
        ),

        # RealSense D435i Camera
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            # Optional: Pass launch arguments to enable the built-in IMU on the D435i
            launch_arguments={
                'enable_gyro': 'true', 
                'enable_accel': 'true',
                'unite_imu_method': '2',  # Combine gyro and accel on /camera/camera/imu.
                'align_depth.enable': 'true',
                'rgb_camera.color_profile': '640x480x15',
                'depth_module.depth_profile': '640x480x15'
            }.items()
        ),

        # Keep acquisition failure behavior from the upstream Velodyne launch.
        velodyne_driver,
        velodyne_cloud,
        RegisterEventHandler(OnProcessExit(
            target_action=velodyne_driver,
            on_exit=[EmitEvent(event=Shutdown())]))
    ])
