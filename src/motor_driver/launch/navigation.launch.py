"""Localize on an exported map using Velodyne ICP and run Nav2."""
from pathlib import Path
import math
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)
    share = Path(get_package_share_directory('motor_driver'))
    map_file = Path(value('map')).expanduser().resolve()
    if not map_file.is_file():
        raise ValueError(f'Exported map YAML does not exist: {map_file}')
    with map_file.open() as stream:
        metadata = yaml.safe_load(stream)
    image = Path(metadata['image'])
    if not image.is_absolute():
        image = map_file.parent / image
    if not image.is_file():
        raise ValueError(f'Map image referenced by YAML does not exist: {image}')
    config = str(share / 'config/nav2.yaml')
    height, yaw = float(value('lidar_height')), float(value('lidar_yaw'))
    if not math.isfinite(height) or height <= 0 or not math.isfinite(yaw):
        raise ValueError('lidar_height must be positive and lidar_yaw finite')
    actions = []
    if value('start_robot') == 'true':
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(share / 'launch/robot_startup.launch.py'))))
    if value('publish_lidar_tf') == 'true':
        actions.append(Node(package='tf2_ros', executable='static_transform_publisher',
                            name='lidar_mount_tf', arguments=[
                                '--x', '0', '--y', '0', '--z', str(height),
                                '--roll', '0', '--pitch', '0', '--yaw', str(yaw),
                                '--frame-id', 'base_link', '--child-frame-id', value('lidar_frame')]))
    core = {'Reg/Strategy': '1', 'Reg/Force3DoF': 'true',
            'Icp/PointToPlane': 'true', 'Icp/VoxelSize': '.10', 'Icp/Iterations': '10',
            'Icp/MaxCorrespondenceDistance': '1', 'Icp/RangeMin': '.5',
            'Icp/RangeMax': '12', 'Icp/CorrespondenceRatio': '.1',
            'Icp/MaxTranslation': '.5', 'OdomF2M/ScanMaxSize': '15000',
            'OdomF2M/ScanSubtractRadius': '.10', 'Odom/ResetCountdown': '0'}
    params = {k: ParameterValue(v, value_type=str) for k, v in core.items()}
    params.update(frame_id='base_link', odom_frame_id='odom', publish_tf=True,
                  wait_for_transform=.3, qos=2, expected_update_rate=10.)
    actions.extend([
        Node(package='rtabmap_odom', executable='icp_odometry', name='icp_odometry',
             output='screen', parameters=[params],
             remappings=[('scan_cloud', value('lidar_topic'))]),
        Node(package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
             name='scan_converter', output='screen', parameters=[config],
             remappings=[('cloud_in', value('lidar_topic')), ('scan', '/scan')]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
            Path(get_package_share_directory('nav2_bringup')) / 'launch/localization_launch.py')),
            launch_arguments={'map': str(map_file), 'params_file': config,
                              'use_sim_time': 'false', 'autostart': 'true',
                              'use_composition': 'False'}.items()),
    ])
    # Use just the servers needed here, independent of distro bringup additions.
    servers = [('nav2_planner', 'planner_server'), ('nav2_controller', 'controller_server'),
               ('nav2_smoother', 'smoother_server'), ('nav2_behaviors', 'behavior_server'),
               ('nav2_bt_navigator', 'bt_navigator'), ('nav2_waypoint_follower', 'waypoint_follower')]
    output = '/navigation/cmd_vel' if value('drive_enabled') == 'true' else '/navigation/cmd_vel_preview'
    for package, executable in servers:
        actions.append(Node(package=package, executable=executable, name=executable,
                            output='screen', parameters=[config],
                            remappings=[('cmd_vel', output)]))
    actions.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                        name='lifecycle_manager_navigation', output='screen',
                        parameters=[{'autostart': True, 'use_sim_time': False,
                                     'node_names': [name for _, name in servers]}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=str(
            Path(get_package_share_directory('motor_driver')) / 'maps/room.yaml'),
                              description='Exported occupancy map YAML and its image.'),
        DeclareLaunchArgument('start_robot', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('drive_enabled', default_value='false', choices=['true', 'false'],
                              description='Forward Nav2 output to the router; GUI Nav2 mode also required.'),
        DeclareLaunchArgument('lidar_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('lidar_frame', default_value='velodyne'),
        DeclareLaunchArgument('lidar_height', default_value='.95'),
        DeclareLaunchArgument('lidar_yaw', default_value='0'),
        DeclareLaunchArgument('publish_lidar_tf', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=launch_setup),
    ])
