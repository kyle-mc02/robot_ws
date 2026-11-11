"""Brown/black rectangular-face RGB-D detection; independent of motor control.

Box axes: x along the 42 cm face edge, y along the 32 cm edge, z into the
5.5 cm thickness. Pose origin is the box center. A symmetric unmarked face
has a 180-degree ambiguity; this is not a uniquely identified object frame.
"""
import math
import heapq

import cv2
import numpy as np
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
    """Return image corners, centroid, and (local mask, x offset, y offset).

    Brown-first segmentation; dark stripes are allowed, never required.
    """
    # Work at <=320 px width rather than running Canny across every full RGB frame.
    scale = min(1.0, 320.0 / bgr.shape[1])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    # Reconnect brown regions split by black lines; fill internal stripe gaps.
    mask = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked = ((cv2.contourArea(contour), index) for index, contour in enumerate(contours))
    for area, index in heapq.nlargest(10, ranked, key=lambda item: item[0]):
        contour = contours[index]
        if area < minimum_area * scale**2 or area > small.shape[0] * small.shape[1] * 0.9:
            continue
        hull = cv2.convexHull(contour)
        quad = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.int32).reshape(4, 1, 2)
        x, y, w, h = cv2.boundingRect(quad)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(mask.shape[1], x+w), min(mask.shape[0], y+h)
        if x1 <= x0 or y1 <= y0:
            continue
        region = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillConvexPoly(region, quad - np.array([x0, y0], np.int32), 255)
        if np.count_nonzero((brown[y0:y1, x0:x1] > 0) & (region > 0)) < 0.25 * np.count_nonzero(region):
            continue
        corners = quad.reshape(4, 2).astype(np.float64) / scale
        moments = cv2.moments(corners.astype(np.float32))
        if abs(moments['m00']) < 1e-8:
            continue
        pixel = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
        angles = np.arctan2(corners[:, 1]-pixel[1], corners[:, 0]-pixel[0])
        corners = corners[np.argsort(angles)]
        corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)
        integer_corners = np.rint(corners).astype(np.int32)
        x, y, w, h = cv2.boundingRect(integer_corners)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(bgr.shape[1], x+w), min(bgr.shape[0], y+h)
        if x1 <= x0 or y1 <= y0:
            continue
        polygon = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillConvexPoly(polygon, integer_corners - np.array([x0, y0], np.int32), 255)
        # Keep image-space corners/center, but allocate only a target-local mask.
        return corners, pixel, (polygon, x0, y0)
    return None


def estimate_target(bgr, depth, k, distortion, dimensions, hsv_low, hsv_high,
                    minimum_area=1200., depth_tolerance=0.04, reprojection_limit=4., detection=None,
                    warn=None):
    """Estimate a pose; report rejection details through an optional warning callback."""
    def reject(reason):
        if warn is not None:
            warn(f'Pose rejected: {reason}')
        return None

    detection = detection if detection is not None else detect_face(bgr, hsv_low, hsv_high, minimum_area)
    if detection is None:
        return reject('no brown rectangular face detected')
    corners, pixel, (polygon, x0, y0) = detection
    depth_roi = depth[y0:y0+polygon.shape[0], x0:x0+polygon.shape[1]]
    u, v = np.rint(pixel).astype(int)
    u, v = u-x0, v-y0
    patch = depth_roi[max(0, v-3):v+4, max(0, u-3):u+4]
    mask = polygon[max(0, v-3):v+4, max(0, u-3):u+4] > 0
    valid = mask & np.isfinite(patch) & (patch > 0.1) & (patch < 10.)
    if np.count_nonzero(valid) < 5:
        return reject(f'center depth patch has {np.count_nonzero(valid)} valid pixels; '
                      'need >=5 finite depths between 0.1 and 10 m')
    z = float(np.median(patch[valid]))
    surface = deproject(pixel, z, k, distortion)
    width, height, thickness = dimensions
    object_points = np.array([[-width/2, -height/2, 0], [width/2, -height/2, 0],
                              [width/2, height/2, 0], [-width/2, height/2, 0]], np.float64)
    # Interior RGB-D surface samples disambiguate planar PnP solutions.
    eroded = cv2.erode(polygon, np.ones((7, 7), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Validate at most 32x32 evenly distributed candidates instead of scanning
    # every depth pixel and building potentially huge coordinate arrays.
    grid_x = np.linspace(0, polygon.shape[1]-1, min(32, polygon.shape[1]), dtype=int)
    grid_y = np.linspace(0, polygon.shape[0]-1, min(32, polygon.shape[0]), dtype=int)
    xx, yy = np.meshgrid(grid_x, grid_y)
    candidate_z = depth_roi[yy, xx]
    valid = (eroded[yy, xx] > 0) & np.isfinite(candidate_z) & (candidate_z > 0.1) & (candidate_z < 10.)
    xx, yy = xx[valid], yy[valid]
    if len(xx) < 20:
        return reject(f'interior depth grid has {len(xx)} valid samples; need >=20 '
                      f'(checked {candidate_z.size} grid locations)')
    indices = np.linspace(0, len(xx)-1, min(150, len(xx)), dtype=int)
    samples_uv = np.column_stack([xx[indices]+x0, yy[indices]+y0]).astype(np.float64)
    rays = cv2.undistortPoints(samples_uv.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    sample_z = depth_roi[yy[indices], xx[indices]]
    samples = np.column_stack([rays * sample_z[:, None], sample_z])
    best = None
    rejections = []
    for shift in range(2):  # 180-degree alternatives are symmetry-equivalent.
        image_points = np.ascontiguousarray(np.roll(corners, shift, axis=0))
        result = cv2.solvePnPGeneric(object_points, image_points, k, distortion, flags=cv2.SOLVEPNP_IPPE)
        if not result[0]:
            rejections.append(f'corner shift {shift}: PnP returned no solutions')
            continue
        for candidate, (rvec, tvec) in enumerate(zip(result[1], result[2])):
            label = f'shift {shift}, candidate {candidate}'
            rotation, _ = cv2.Rodrigues(rvec)
            translation = tvec.ravel()
            normal = rotation[:, 2]
            if translation[2] <= 0 or np.dot(normal, translation) <= 0:
                rejections.append(f'{label}: invalid face direction / behind camera '
                                  f'(z={translation[2]:.3f} m, normal dot position={np.dot(normal, translation):.3f})')
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-image_points)**2, axis=1))))
            plane_error = float(np.median(np.abs((samples-translation) @ normal)))
            if error > reprojection_limit or plane_error > depth_tolerance:
                rejections.append(f'{label}: reprojection={error:.2f} px '
                                  f'(limit {reprojection_limit:.2f}), plane depth error={plane_error:.4f} m '
                                  f'(limit {depth_tolerance:.4f})')
                continue
            center_error = float(abs(np.dot(surface-translation, normal)))
            if center_error > depth_tolerance:
                rejections.append(f'{label}: center depth error={center_error:.4f} m '
                                  f'(limit {depth_tolerance:.4f})')
                continue
            # Stable image-corner convention breaks indistinguishable symmetry ties.
            score = error + 100 * plane_error
            if best is None or score < best[0] - 1e-6:
                best = (score, translation + normal * thickness/2, rotation)
    if best is None:
        return reject('no valid PnP pose; ' + '; '.join(rejections))
    normal = best[2][:, 2]
    # Refine range with measured depth after validating the known face dimensions.
    # Brown segmentation can sit just inside a dark border; do not let that
    # small corner-size error bias the otherwise directly measured depth.
    face_center = best[1] - normal * thickness/2
    depth_correction = normal * np.median((samples-face_center) @ normal)
    best = (best[0], best[1] + depth_correction, best[2])
    face_center = best[1] - normal * thickness/2
    corner_rays = cv2.undistortPoints(corners.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    corner_rays = np.column_stack([corner_rays, np.ones(4)])
    denominator = corner_rays @ normal
    if np.any(np.abs(denominator) < 1e-6):
        return reject('corner viewing ray is nearly parallel to the estimated face plane')
    corner_depths = np.dot(face_center, normal) / denominator
    if np.any(corner_depths <= 0):
        return reject(f'estimated face corners lie behind camera: depths={corner_depths.tolist()} m')
    corners_camera = corner_rays * corner_depths[:, None]
    return {'corners_camera': corners_camera, 'pixel': pixel, 'surface': surface, 'center': best[1], 'rotation': best[2], 'corners': corners}


class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        defaults = {
            'rgb_topic': '/camera/camera/color/image_raw/compressed',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'info_topic': '/camera/camera/color/camera_info',
            'base_frame': 'base_link', 'mount_frame': 'camera_link',
            'publish_mount_tf': True, 'mount_xyz': [0.24, -0.01, 0.55],
            'mount_rpy': [0.0, 0.0, 0.0],
            'box_dimensions': [0.42, 0.32, 0.055],
            'brown_hsv_low': [0, 25, 20], 'brown_hsv_high': [40, 255, 255],
            'min_contour_area': 1200.0,
            'processing_rate_hz': 3.0,
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
        for name in ('min_contour_area', 'depth_tolerance', 'max_reprojection_error'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        rate = p['processing_rate_hz']
        if not math.isfinite(rate) or not 0 < rate <= 15:
            raise ValueError('processing_rate_hz must be positive and <= 15')
        cv2.setNumThreads(1)  # Avoid competing with camera encoding across CPU cores.
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
        self.point_camera_pub = self.create_publisher(PointStamped, '/perception/point_camera', qos_profile_sensor_data)
        self.point_base_pub = self.create_publisher(PointStamped, '/perception/point_base', qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, '/perception/box_pose', qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/markers', qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(CompressedImage, '/perception/debug/compressed', qos_profile_sensor_data)
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.rgb_sub = self.create_subscription(CompressedImage, p['rgb_topic'], self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, p['depth_topic'], self.depth_callback, qos_profile_sensor_data)
        self.info_sub = self.create_subscription(CameraInfo, p['info_topic'], self.info_callback, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self.process_latest)
        self.last_warning = -float('inf')
        self.get_logger().info(
            f'Perception at up to {rate:g} Hz; run on the robot so raw RGB/depth stay local.')

    def info_callback(self, info):
        self.camera_info = info

    def rgb_callback(self, rgb):
        self.latest_rgb = rgb

    def depth_callback(self, depth):
        self.latest_depth = depth

    def process_latest(self):
        rgb, self.latest_rgb = self.latest_rgb, None
        if rgb is None:
            return
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(rgb, 'bgr8')
            detection = detect_face(bgr, self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                    self.values['min_contour_area'])
            info = self.camera_info
            if detection is None:
                self.marker_pub.publish(MarkerArray(markers=[Marker(
                    ns='detection_rays', id=4, action=Marker.DELETE)]))
                return
            if info is None or info.header.frame_id != rgb.header.frame_id:
                self.warn('Brown region detected; waiting for matching color CameraInfo')
                return
            k = np.asarray(info.k, np.float64).reshape(3, 3)
            if (info.width, info.height) != (bgr.shape[1], bgr.shape[0]) or k[0, 0] <= 0 or k[1, 1] <= 0:
                self.warn('Brown region detected; CameraInfo dimensions/intrinsics mismatch')
                return
            # Viewing rays encode the 2D detection without claiming a 3D location.
            # RViz can display these rays; the GUI projects their endpoints to RGB.
            pixels = np.vstack([detection[0], detection[1]])
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, np.asarray(info.d, np.float64)).reshape(-1, 2)
            rays = Marker(header=rgb.header, ns='detection_rays', id=4,
                          type=Marker.LINE_LIST, action=Marker.ADD)
            rays.pose.orientation.w = 1.
            rays.scale.x = 0.002
            rays.color.g, rays.color.a = 1., 0.15
            rays.lifetime.nanosec = 600000000
            rays.points = []
            for x, y in xy:
                rays.points.extend([Point(), Point(x=float(x), y=float(y), z=1.)])
            self.marker_pub.publish(MarkerArray(markers=[rays]))
            depth = self.latest_depth
            if depth is None or abs(self.stamp_seconds(depth.header)-self.stamp_seconds(rgb.header)) > 0.1:
                self.warn('2D detection active; aligned depth is missing or older than 0.1 s')
                return
            self.process(rgb, depth, info, bgr=bgr, detection=detection)
        except (ValueError, CvBridgeError, cv2.error) as error:
            self.warn(str(error))

    @staticmethod
    def stamp_seconds(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def warn(self, message):
        now = self.get_clock().now().nanoseconds / 1e9
        if now-self.last_warning >= 2:
            self.get_logger().warning(message)
            self.last_warning = now

    def process(self, rgb, depth_msg, info, bgr=None, detection=None):
        try:
            if bgr is None:
                bgr = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
            if not rgb.header.frame_id or info.header.frame_id != rgb.header.frame_id or depth_msg.header.frame_id != rgb.header.frame_id:
                raise ValueError('RGB, aligned depth and CameraInfo must share the color optical frame')
            if (bgr.shape[1], bgr.shape[0]) != (depth_msg.width, depth_msg.height) or (info.width, info.height) != (bgr.shape[1], bgr.shape[0]):
                raise ValueError('RGB, aligned depth and CameraInfo dimensions must match')
            if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                raise ValueError('Unsupported distortion model; use calibrated plumb_bob/rational_polynomial RGB')
            k = np.asarray(info.k, np.float64).reshape(3, 3)
            distortion = np.asarray(info.d, np.float64)
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(distortion).all():
                raise ValueError('Invalid camera intrinsics')
            depth = depth_in_meters(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'), depth_msg.encoding)
            if detection is None:
                detection = detect_face(bgr, self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                        self.values['min_contour_area'])
            if detection is None:
                self.warn('No brown rectangular target found; check lighting and brown_hsv_low/high')
                return
            result = estimate_target(bgr, depth, k, distortion, self.dimensions,
                                     self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                     self.values['min_contour_area'], self.values['depth_tolerance'],
                                     self.values['max_reprojection_error'], detection=detection,
                                     warn=self.warn)
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
