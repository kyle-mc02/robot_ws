"""Separate LiDAR or RGB-D RTAB-Map sessions using existing robot sensors."""
from datetime import datetime
from pathlib import Path
import math

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

    sensor = value('sensor')
    share = Path(get_package_share_directory('motor_driver'))
    viewer = Node(package='rviz2', executable='rviz2', name='slam_view', output='screen')
    if value('view_only') == 'true':
        return [viewer]

    database = value('database_path') or str(
        Path.home()/'.ros/victor_slam'/f'{sensor}_{datetime.now():%Y%m%d_%H%M%S_%f}.db')
    database = Path(database).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    actions = []
    if value('start_robot') == 'true':
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(share/'launch/robot_startup.launch.py'))))
    if sensor == 'lidar' and value('publish_lidar_tf') == 'true':
        height = float(value('lidar_height'))
        yaw = float(value('lidar_yaw'))
        if not math.isfinite(height) or height <= 0 or not math.isfinite(yaw):
            raise ValueError('lidar_height must be positive; lidar_yaw must be finite (radians)')
        actions.append(Node(package='tf2_ros', executable='static_transform_publisher',
                            name='lidar_mount_tf', arguments=[
                                '--x', '0', '--y', '0', '--z', str(height),
                                '--roll', '0', '--pitch', '0', '--yaw', str(yaw),
                                '--frame-id', 'base_link', '--child-frame-id', value('lidar_frame')]))

    if sensor == 'rgbd':
        # Perception is disabled in robot startup; own only its camera mount TF.
        actions.append(Node(package='tf2_ros', executable='static_transform_publisher',
                            name='camera_mount_tf', arguments=[
                                '--x', '.24', '--y', '-.01', '--z', '.55',
                                '--roll', '0', '--pitch', '0', '--yaw', '0',
                                '--frame-id', 'base_link', '--child-frame-id', 'camera_link']))

    # Planar robot motion with 3D sensor data retained for cloud-map export.
    common = {'frame_id': 'base_link', 'wait_for_transform': .3,
              'qos': 2, 'Reg/Force3DoF': 'true'}
    mapping = {'database_path': str(database), 'map_frame_id': 'map',
               'subscribe_depth': sensor == 'rgbd', 'subscribe_rgb': sensor == 'rgbd',
               'subscribe_scan_cloud': sensor == 'lidar', 'subscribe_odom_info': True,
               'approx_sync': True, 'approx_sync_max_interval': .1,
               'topic_queue_size': 5, 'sync_queue_size': 5,
               'qos_image': 2, 'qos_camera_info': 2, 'qos_scan': 2, 'qos_odom': 2,
               'RGBD/CreateOccupancyGrid': 'true', 'Rtabmap/DetectionRate': '1',
               'RGBD/LinearUpdate': '.10', 'RGBD/AngularUpdate': '.10',
               'Grid/Sensor': '1' if sensor == 'rgbd' else '0',
               'Grid/3D': 'true', 'Grid/CellSize': '.05', 'Grid/RayTracing': 'true',
               'Grid/NormalsSegmentation': 'false', 'Grid/MinGroundHeight': '-0.15',
               'Grid/MaxGroundHeight': '.08', 'Grid/MaxObstacleHeight': '1.8',
               'Grid/RangeMin': '.35', 'Grid/RangeMax': '4' if sensor == 'rgbd' else '12',
               'Grid/DepthDecimation': '4', 'cloud_output_voxelized': True}
    remaps = [('scan_cloud', value('lidar_topic')),
              ('rgb/image', '/camera/camera/color/image_raw'),
              ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
              ('rgb/camera_info', '/camera/camera/color/camera_info')]
    if sensor == 'lidar':
        common.update({'Reg/Strategy': '1', 'Icp/PointToPlane': 'true',
                       'Icp/VoxelSize': '.10', 'Icp/Iterations': '10',
                       'Icp/MaxCorrespondenceDistance': '1', 'Icp/RangeMin': '.5',
                       'Icp/RangeMax': '12', 'Icp/CorrespondenceRatio': '.1'})
        odometry = {'odom_frame_id': 'odom', 'publish_tf': True,
                    'expected_update_rate': 15., 'OdomF2M/ScanMaxSize': '15000',
                    'OdomF2M/ScanSubtractRadius': '.10'}
        mapping.update({'RGBD/ProximityBySpace': 'true', 'RGBD/ProximityPathMaxNeighbors': '1',
                        'RGBD/ProximityMaxGraphDepth': '0', 'Icp/CorrespondenceRatio': '.2'})
        executable = 'icp_odometry'
    else:
        common['Reg/Strategy'] = '0'
        odometry = {'odom_frame_id': 'odom', 'publish_tf': True,
                    'approx_sync': True, 'approx_sync_max_interval': .1,
                    'topic_queue_size': 5, 'sync_queue_size': 5,
                    'qos_camera_info': 2, 'Vis/MaxDepth': '4'}
        executable = 'rgbd_odometry'
    # RTAB-Map core parameters are strings, even when they look numeric.
    def typed(parameters):
        return {key: ParameterValue(val, value_type=str) if '/' in key else val
                for key, val in parameters.items()}

    actions.extend([
        Node(package='rtabmap_odom', executable=executable, name=executable,
             namespace='slam', output='screen', parameters=[typed(common), typed(odometry)], remappings=remaps),
        Node(package='rtabmap_slam', executable='rtabmap', name='rtabmap',
             namespace='slam', output='screen', parameters=[typed(common), typed(mapping)], remappings=remaps)])
    if value('rviz') == 'true':
        actions.append(viewer)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sensor', default_value='lidar', choices=['lidar', 'rgbd'],
                              description='Independent mapping run; never launch both modes together.'),
        DeclareLaunchArgument('start_robot', default_value='false', choices=['true', 'false'],
                              description='Include robot_startup only when it is not already running.'),
        DeclareLaunchArgument('view_only', default_value='false', choices=['true', 'false'],
                              description='Basestation RViz only; no sensor or mapping nodes.'),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('database_path', default_value='',
                              description='Empty creates a unique persistent database for each run.'),
        DeclareLaunchArgument('lidar_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('lidar_frame', default_value='velodyne'),
        DeclareLaunchArgument('lidar_height', default_value='.95'),
        DeclareLaunchArgument('lidar_yaw', default_value='0',
                              description='Sensor x-axis yaw relative to robot forward, in radians.'),
        DeclareLaunchArgument('publish_lidar_tf', default_value='true', choices=['true', 'false'],
                              description='Disable if another publisher already owns this mount TF.'),
        OpaqueFunction(function=launch_setup)])
