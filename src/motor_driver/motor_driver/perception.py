"""Brown/black rectangular-face RGB-D detection; independent of motor control.

Box axes: x along the 42 cm face edge, y along the 32 cm edge, z into the
5.5 cm thickness. Pose origin is the box center. A symmetric unmarked face
has a 180-degree ambiguity; this is not a uniquely identified object frame.
"""
import math

import cv2
import numpy as np
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException
from tf2_geometry_msgs import do_transform_point, do_transform_pose


def quaternion_from_matrix(rotation):
    """Unit quaternion xyzw, including rotations close to 180 degrees."""
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    return np.r_[rvec.ravel() / angle * math.sin(angle / 2), math.cos(angle / 2)]


def depth_in_meters(depth, encoding):
    if encoding == '16UC1':
        return depth.astype(np.float64) * 0.001
    if encoding == '32FC1':
        return depth.astype(np.float64)
    raise ValueError('Depth must be 16UC1 millimeters or 32FC1 meters')


def deproject(pixel, z, k, distortion):
    ray = cv2.undistortPoints(np.asarray(pixel, np.float64).reshape(1, 1, 2), k, distortion)[0, 0]
    return np.array([ray[0] * z, ray[1] * z, z])


def detect_face(bgr, hsv_low, hsv_high, minimum_area):
    """Find a convex four-corner boundary with brown fill and black markings."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    black = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8), np.array([179, 255, 75], np.uint8))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < minimum_area or area > bgr.shape[0] * bgr.shape[1] * 0.9:
            continue
        quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        polygon = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(polygon, quad, 255)
        count = np.count_nonzero(polygon)
        brown_fraction = np.count_nonzero((brown > 0) & (polygon > 0)) / count
        black_fraction = np.count_nonzero((black > 0) & (polygon > 0)) / count
        if brown_fraction < 0.12 or black_fraction < 0.02 or brown_fraction + black_fraction < 0.55:
            continue
        moments = cv2.moments(quad)
        pixel = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
        corners = quad.reshape(4, 2).astype(np.float64)
        # Clockwise in image coordinates, starting at the top-left convention.
        angles = np.arctan2(corners[:, 1]-pixel[1], corners[:, 0]-pixel[0])
        corners = corners[np.argsort(angles)]
        corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)
        return corners, pixel, polygon
    return None


def estimate_target(bgr, depth, k, distortion, dimensions, hsv_low, hsv_high,
                    minimum_area=1200., depth_tolerance=0.04, reprojection_limit=4.):
    detection = detect_face(bgr, hsv_low, hsv_high, minimum_area)
    if detection is None:
        return None
    corners, pixel, polygon = detection
    u, v = np.rint(pixel).astype(int)
    patch = depth[max(0, v-3):v+4, max(0, u-3):u+4]
    mask = polygon[max(0, v-3):v+4, max(0, u-3):u+4] > 0
    valid = mask & np.isfinite(patch) & (patch > 0.1) & (patch < 10.)
    if np.count_nonzero(valid) < 5:
        return None
    z = float(np.median(patch[valid]))
    surface = deproject(pixel, z, k, distortion)
    width, height, thickness = dimensions
    object_points = np.array([[-width/2, -height/2, 0], [width/2, -height/2, 0],
                              [width/2, height/2, 0], [-width/2, height/2, 0]], np.float64)
    # Interior RGB-D surface samples disambiguate planar PnP solutions.
    eroded = cv2.erode(polygon, np.ones((7, 7), np.uint8))
    yy, xx = np.nonzero((eroded > 0) & np.isfinite(depth) & (depth > 0.1) & (depth < 10.))
    if len(xx) < 20:
        return None
    indices = np.linspace(0, len(xx)-1, min(150, len(xx)), dtype=int)
    samples_uv = np.column_stack([xx[indices], yy[indices]]).astype(np.float64)
    rays = cv2.undistortPoints(samples_uv.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    sample_z = depth[yy[indices], xx[indices]]
    samples = np.column_stack([rays * sample_z[:, None], sample_z])
    best = None
    for shift in range(4):
        image_points = np.ascontiguousarray(np.roll(corners, shift, axis=0))
        result = cv2.solvePnPGeneric(object_points, image_points, k, distortion, flags=cv2.SOLVEPNP_IPPE)
        if not result[0]:
            continue
        for rvec, tvec in zip(result[1], result[2]):
            rotation, _ = cv2.Rodrigues(rvec)
            translation = tvec.ravel()
            normal = rotation[:, 2]
            if translation[2] <= 0 or np.dot(normal, translation) <= 0:
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-image_points)**2, axis=1))))
            plane_error = float(np.median(np.abs((samples-translation) @ normal)))
            if error > reprojection_limit or plane_error > depth_tolerance:
                continue
            if abs(np.dot(surface-translation, normal)) > depth_tolerance:
                continue
            # Stable image-corner convention breaks indistinguishable symmetry ties.
            score = error + 100 * plane_error
            if best is None or score < best[0] - 1e-6:
                best = (score, translation + normal * thickness/2, rotation)
    if best is None:
        return None
    normal = best[2][:, 2]
    face_center = best[1] - normal * thickness/2
    corner_rays = cv2.undistortPoints(corners.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    corner_rays = np.column_stack([corner_rays, np.ones(4)])
    denominator = corner_rays @ normal
    if np.any(np.abs(denominator) < 1e-6):
        return None
    corner_depths = np.dot(face_center, normal) / denominator
    if np.any(corner_depths <= 0):
        return None
    corners_camera = corner_rays * corner_depths[:, None]
    return {'corners_camera': corners_camera, 'pixel': pixel, 'surface': surface, 'center': best[1], 'rotation': best[2], 'corners': corners}


class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        defaults = {
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'info_topic': '/camera/camera/color/camera_info',
            'base_frame': 'base_link', 'mount_frame': 'camera_link',
            'publish_mount_tf': True, 'mount_xyz': [0.24, -0.01, 0.55],
            'mount_rpy': [0.0, 0.0, 0.0],
            'box_dimensions': [0.42, 0.32, 0.055],
            'brown_hsv_low': [5, 40, 25], 'brown_hsv_high': [35, 255, 255],
            'min_contour_area': 1200.0, 'sync_slop': 0.04,
            'depth_tolerance': 0.04, 'max_reprojection_error': 4.0}
        self.values = {}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            self.values[name] = self.get_parameter(name).value
        p = self.values
        self.dimensions = np.asarray(p['box_dimensions'], np.float64)
        if self.dimensions.shape != (3,) or not np.isfinite(self.dimensions).all() or np.any(self.dimensions <= 0):
            raise ValueError('box_dimensions must be three positive finite meters')
        for name in ('mount_xyz', 'mount_rpy'):
            if len(p[name]) != 3 or not np.isfinite(p[name]).all():
                raise ValueError(f'{name} must contain three finite values')
        if p['base_frame'] == p['mount_frame']:
            raise ValueError('base_frame and mount_frame must differ')
        for name, maximum in [('brown_hsv_low', [179, 255, 255]), ('brown_hsv_high', [179, 255, 255])]:
            if len(p[name]) != 3 or np.any(np.array(p[name]) < 0) or np.any(np.array(p[name]) > maximum):
                raise ValueError(f'Invalid {name}')
        if np.any(np.array(p['brown_hsv_low']) > p['brown_hsv_high']):
            raise ValueError('HSV lower bound must not exceed upper bound')
        for name in ('min_contour_area', 'sync_slop', 'depth_tolerance', 'max_reprojection_error'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.bridge = CvBridge()
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        if p['publish_mount_tf']:
            self.broadcaster = StaticTransformBroadcaster(self)
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id, transform.child_frame_id = p['base_frame'], p['mount_frame']
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, p['mount_xyz'])
            roll, pitch, yaw = p['mount_rpy']
            cr, sr, cp, sp, cy, sy = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2), math.cos(yaw/2), math.sin(yaw/2)
            q = transform.transform.rotation
            q.x, q.y, q.z, q.w = sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy
            self.broadcaster.sendTransform(transform)
        self.point_camera_pub = self.create_publisher(PointStamped, '/perception/point_camera', 10)
        self.point_base_pub = self.create_publisher(PointStamped, '/perception/point_base', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/perception/box_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/markers', 10)
        self.debug_pub = self.create_publisher(CompressedImage, '/perception/debug/compressed', qos_profile_sensor_data)
        self.rgb_sub = message_filters.Subscriber(self, Image, p['rgb_topic'], qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, p['depth_topic'], qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, p['info_topic'], qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], queue_size=5, slop=p['sync_slop'])
        self.sync.registerCallback(self.process)
        self.last_warning = -float('inf')

    def warn(self, message):
        now = self.get_clock().now().nanoseconds / 1e9
        if now-self.last_warning >= 2:
            self.get_logger().warning(message)
            self.last_warning = now

    def process(self, rgb, depth_msg, info):
        try:
            if not rgb.header.frame_id or info.header.frame_id != rgb.header.frame_id or depth_msg.header.frame_id != rgb.header.frame_id:
                raise ValueError('RGB, aligned depth and CameraInfo must share the color optical frame')
            if (rgb.width, rgb.height) != (depth_msg.width, depth_msg.height) or (info.width, info.height) != (rgb.width, rgb.height):
                raise ValueError('RGB, aligned depth and CameraInfo dimensions must match')
            if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                raise ValueError('Unsupported distortion model; use calibrated plumb_bob/rational_polynomial RGB')
            k = np.asarray(info.k, np.float64).reshape(3, 3)
            distortion = np.asarray(info.d, np.float64)
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(distortion).all():
                raise ValueError('Invalid camera intrinsics')
            bgr = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
            depth = depth_in_meters(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'), depth_msg.encoding)
            result = estimate_target(bgr, depth, k, distortion, self.dimensions,
                                     self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                     self.values['min_contour_area'], self.values['depth_tolerance'],
                                     self.values['max_reprojection_error'])
            if result is None:
                return  # RViz markers expire after 0.5 s; no stale pose is republished.
            point = PointStamped(header=rgb.header)
            point.point.x, point.point.y, point.point.z = map(float, result['surface'])
            self.point_camera_pub.publish(point)
            if self.debug_pub.get_subscription_count():
                cv2.polylines(bgr, [result['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                u, v = np.rint(result['pixel']).astype(int)
                cv2.circle(bgr, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(bgr, f'center ({u}, {v}) px; depth {result["surface"][2]:.2f}m',
                            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                debug = self.bridge.cv2_to_compressed_imgmsg(bgr)
                debug.header = rgb.header
                self.debug_pub.publish(debug)
            try:
                transform = self.buffer.lookup_transform(self.values['base_frame'], rgb.header.frame_id, Time.from_msg(rgb.header.stamp))
            except TransformException as error:
                self.warn(f'Camera point detected, but base TF unavailable: {error}')
                return
            base_point = do_transform_point(point, transform)
            base_point.header.stamp = rgb.header.stamp
            self.point_base_pub.publish(base_point)
            pose = PoseStamped(header=base_point.header)
            camera_pose = PoseStamped()
            camera_pose.pose.position.x, camera_pose.pose.position.y, camera_pose.pose.position.z = map(float, result['center'])
            q = quaternion_from_matrix(result['rotation'])
            camera_pose.pose.orientation.x, camera_pose.pose.orientation.y, camera_pose.pose.orientation.z, camera_pose.pose.orientation.w = map(float, q)
            pose.pose = do_transform_pose(camera_pose.pose, transform)
            self.pose_pub.publish(pose)
            self.publish_markers(base_point, pose, rgb.header, result['corners_camera'], result['center'])
        except (ValueError, CvBridgeError, cv2.error) as error:
            self.warn(str(error))

    def publish_markers(self, point, pose, camera_header, corners_camera, center_camera):
        sphere = Marker(header=point.header, ns='detected_surface', id=0, type=Marker.SPHERE, action=Marker.ADD)
        sphere.pose.position = point.point
        sphere.pose.orientation.w = 1.
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.035
        sphere.color.g, sphere.color.a = 1., 1.
        cube = Marker(header=pose.header, ns='box_pose', id=1, type=Marker.CUBE, action=Marker.ADD)
        cube.pose = pose.pose
        cube.scale.x, cube.scale.y, cube.scale.z = map(float, self.dimensions)
        cube.color.b, cube.color.a = 1., 0.35
        contour = Marker(header=camera_header, ns='detected_contour', id=2,
                         type=Marker.LINE_STRIP, action=Marker.ADD)
        contour.pose.orientation.w = 1.
        contour.scale.x = 0.004
        contour.color.g, contour.color.a = 1., 1.
        contour.points = [Point(x=float(x), y=float(y), z=float(z))
                          for x, y, z in np.vstack([corners_camera, corners_camera[:1]])]
        center = Marker(header=camera_header, ns='box_center_camera', id=3,
                        type=Marker.SPHERE, action=Marker.ADD)
        center.pose.position.x, center.pose.position.y, center.pose.position.z = map(float, center_camera)
        center.pose.orientation.w = 1.
        center.scale.x = center.scale.y = center.scale.z = 0.01
        center.color.r, center.color.g, center.color.a = 1., 1., 1.
        for marker in (sphere, cube, contour, center):
            marker.lifetime.sec, marker.lifetime.nanosec = 0, 500000000
        self.marker_pub.publish(MarkerArray(markers=[sphere, cube, contour, center]))


def main(args=None):
    rclpy.init(args=args)
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
