"""Brown/black rectangular-face RGB-D detection; independent of motor control.

Box axes: x along the 42 cm face edge, y along the 32 cm edge, z into the
5.5 cm thickness. Pose origin is the box center. A symmetric unmarked face
has a 180-degree ambiguity; this is not a uniquely identified object frame
(a temporal filter picks the 180-degree alternative closest to the last pose).

Pipeline per frame:
  1. detect_faces: HSV brown seeds, dark border growth limited to pixels that
     are near the brown AND on the same depth plane as the brown. These are
     only *seed regions*; their 2D size is never trusted.
  2. fit_fixed_rect: fit the lid plane from depth inside the seed, keep every
     pixel on that plane, and slide a fixed 0.42 x 0.32 m template over it.
     Size cannot shrink or grow; only pose is optimised.
  3. Fallback: PnP on the 2D corners if the template fit fails.
  4. PoseFilter: EMA smoothing, jump rejection, 180-degree disambiguation.
"""
import math
import heapq
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException
from tf2_geometry_msgs import do_transform_point, do_transform_pose

HULL_MAX_GROWTH = 1.4   # hull area / brown seed area; real dark borders add 10-30 %
ASPECT_MAX = 2.0        # nominal 42/32 = 1.31
DARK_VALUE = 105        # HSV V below this counts as dark border
NEAR_BROWN_ITERS = 3    # without depth: dark pixels must lie within ~6 px (at 320 px width) of brown
NEAR_BROWN_ITERS_GATED = 8  # with depth the plane gate blocks background, so reach further


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
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


def backproject_roi(depth_roi, x0, y0, k):
    """Pinhole back-projection of a depth crop (aligned depth is already rectified).

    Returns (points HxWx3 with invalid depths set to 0, validity mask)."""
    ok = np.isfinite(depth_roi) & (depth_roi > 0.1) & (depth_roi < 10.)
    z = np.where(ok, depth_roi, 0.)
    v, u = np.mgrid[y0:y0 + depth_roi.shape[0], x0:x0 + depth_roi.shape[1]]
    pts = np.dstack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
    return pts, ok


class FaceDetection(tuple):
    """Image box is always usable; pose corners require a recovered boundary."""
    def __new__(cls, corners, pixel, region, pose_corners):
        result = super().__new__(cls, (corners, pixel, region))
        result.pose_corners = pose_corners
        return result


def ordered_corners(corners):
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    center = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))]
    return np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)


def fit_surface_plane(samples, tolerance):
    """Bounded deterministic RANSAC, then SVD; reject mixed tray/floor surfaces.

    Returns (inlier samples, normal pointing away from the camera, center)."""
    samples = np.asarray(samples, np.float64)
    if len(samples) < 20:
        return None
    rng = np.random.default_rng(5)
    best = np.zeros(len(samples), bool)
    threshold = min(0.015, tolerance / 2)
    for _ in range(32):
        a, b, c = samples[rng.choice(len(samples), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inliers = np.abs((samples - a) @ normal) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(20, int(np.ceil(0.75 * len(samples)))):
        return None
    clean = samples[best]
    center = clean.mean(axis=0)
    _, _, vectors = np.linalg.svd(clean - center, full_matrices=False)
    normal = vectors[-1]
    if normal @ center < 0:
        normal = -normal
    return clean, normal, center


# ----------------------------------------------------------------------------
# Step 1: seed detection (HSV + depth-gated border growth)
# ----------------------------------------------------------------------------
def lid_plane_gate(depth_small, k, scale, brown_roi, x0, y0, tol=0.025):
    """True where ROI pixels lie within tol metres of the plane fitted to brown pixels."""
    h, w = brown_roi.shape
    d = depth_small[y0:y0 + h, x0:x0 + w]
    ok = np.isfinite(d) & (d > 0.1) & (d < 3.0)
    z = np.where(ok, d, 0.)
    v, u = np.mgrid[y0:y0 + h, x0:x0 + w]
    pts = np.dstack([(u / scale - k[0, 2]) * z / k[0, 0],
                     (v / scale - k[1, 2]) * z / k[1, 1], z])
    seed = pts[brown_roi & ok]
    if len(seed) < 50:
        return None
    seed = seed[np.linspace(0, len(seed) - 1, min(300, len(seed)), dtype=int)]
    plane = fit_surface_plane(seed, tol * 2)
    if plane is None:
        return None
    _, normal, center = plane
    return ok & (np.abs((pts - center) @ normal) < tol)


def detect_faces(bgr, hsv_low, hsv_high, minimum_area, depth=None, k=None, limit=3):
    """Yield up to `limit` FaceDetection seeds, largest brown region first.

    HSV locates brown; surrounding dark borders are recovered in a local ROI,
    restricted (when depth is given) to pixels on the brown region's plane."""
    scale = min(1.0, 320.0 / bgr.shape[1])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    depth_small = None
    if depth is not None and k is not None:
        depth_small = cv2.resize(depth, (small.shape[1], small.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    mask = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked = ((cv2.contourArea(c), i) for i, c in enumerate(contours))
    yielded = 0
    for area, index in heapq.nlargest(10, ranked, key=lambda item: item[0]):
        if yielded >= limit:
            return
        if area < minimum_area * scale ** 2 or area > small.shape[0] * small.shape[1] * 0.85:
            continue
        seed = contours[index]
        x, y, w, h = cv2.boundingRect(seed)
        pad = max(5, int(min(w, h) * 0.3))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(small.shape[1], x + w + pad), min(small.shape[0], y + h + pad)
        roi = hsv[y0:y1, x0:x1]
        brown_roi = brown[y0:y1, x0:x1] > 0
        # Dark pixels count only near brown, so far-away shadows cannot join. With depth,
        # the plane gate does the background rejection, so the border may reach further.
        near = cv2.dilate(brown_roi.astype(np.uint8), np.ones((5, 5), np.uint8),
                          iterations=NEAR_BROWN_ITERS_GATED if depth_small is not None
                          else NEAR_BROWN_ITERS) > 0
        grow = brown_roi | ((roi[:, :, 2] < DARK_VALUE) & near)
        if depth_small is not None:
            gate = lid_plane_gate(depth_small, k, scale, brown_roi, x0, y0)
            if gate is None:
                continue  # brown seed is not one flat surface
            grow &= gate
        combined = cv2.morphologyEx(grow.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8))
        local, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seed_center = (float(x + w / 2 - x0), float(y + h / 2 - y0))
        candidates = []
        for contour in local:
            if cv2.pointPolygonTest(contour, seed_center, False) < 0:
                continue
            hull = cv2.convexHull(contour)
            if not 0.85 * area <= cv2.contourArea(hull) <= HULL_MAX_GROWTH * area:
                continue
            for epsilon in (0.015, 0.025, 0.04):
                quad = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
                if len(quad) == 4 and cv2.isContourConvex(quad):
                    q = quad.reshape(4, 2).astype(np.float64)
                    sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
                    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
                    if min(a, b) > 0 and max(a, b) / min(a, b) <= ASPECT_MAX:
                        candidates.append((cv2.contourArea(quad), q + [x0, y0]))
                        break
        pose_corners = None
        if candidates:
            pose_corners = ordered_corners(max(candidates, key=lambda q: q[0])[1] / scale)
            corners = pose_corners
        else:
            # Detection survives irregular contours; never feed synthetic rectangle
            # corners to PnP as if they were physical perspective corners.
            corners = ordered_corners(cv2.boxPoints(cv2.minAreaRect(seed)) / scale)
        pixel = corners.mean(axis=0)
        integer = np.rint(corners).astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(integer)
        rx, ry = max(0, rx), max(0, ry)
        right, bottom = min(bgr.shape[1], rx + rw), min(bgr.shape[0], ry + rh)
        if right <= rx or bottom <= ry:
            continue
        polygon = np.zeros((bottom - ry, right - rx), np.uint8)
        cv2.fillConvexPoly(polygon, integer - np.array([rx, ry], np.int32), 255)
        yielded += 1
        yield FaceDetection(corners, pixel, (polygon, rx, ry), pose_corners)


# ----------------------------------------------------------------------------
# Step 2: fixed-size rectangle fit on the lid plane
# ----------------------------------------------------------------------------
@lru_cache(maxsize=512)
def rect_template(width, height, deci_degrees, resolution, ring=0.03):
    """+1 inside the rectangle, -1 in a thin ring outside (penalises background bleed)."""
    theta = math.radians(deci_degrees / 10.)
    size = (int(np.hypot(width, height) / resolution + 2 * ring / resolution) + 4) | 1
    c = size // 2
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    def poly(ww, hh):
        r = np.array([[-ww / 2, -hh / 2], [ww / 2, -hh / 2], [ww / 2, hh / 2], [-ww / 2, hh / 2]]) / resolution
        return np.rint(r @ R.T + c).astype(np.int32)

    t = np.zeros((size, size), np.float32)
    cv2.fillConvexPoly(t, poly(width + 2 * ring, height + 2 * ring), -1.0)
    cv2.fillConvexPoly(t, poly(width, height), 1.0)
    return t


def make_result(center, rotation, dimensions, k, distortion, score=0.):
    """Build the result dict every downstream consumer uses from a box pose (camera frame)."""
    width, height, thickness = dimensions
    center = np.asarray(center, np.float64)
    rotation = np.asarray(rotation, np.float64)
    normal = rotation[:, 2]
    face = center - normal * thickness / 2
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    corners_camera = (face + signs[:, :1] * rotation[:, 0] * width / 2
                      + signs[:, 1:] * rotation[:, 1] * height / 2)
    if np.any(corners_camera[:, 2] <= 0.05) or face[2] <= 0.05:
        return None
    projected, _ = cv2.projectPoints(np.vstack([corners_camera, face]),
                                     np.zeros(3), np.zeros(3), k, distortion)
    projected = projected.reshape(-1, 2)
    return {'score': float(score), 'corners_camera': corners_camera, 'corners': projected[:4],
            'pixel': projected[4], 'surface': face, 'center': center, 'rotation': rotation}


def fit_fixed_rect(detection, depth, k, distortion, dimensions, plane_tolerance=0.02,
                   min_coverage=0.6, resolution=0.01, warn=None):
    """Fit the known-size face to depth. The 2D detection only seeds the search.

    1. Fit the lid plane to depth inside the (eroded) seed polygon.
    2. Keep all pixels within plane_tolerance of that plane (includes the dark border,
       excludes the surface the box sits on).
    3. Slide the fixed-size template over the top-down occupancy at every yaw."""
    def fail(reason):
        if warn is not None:
            warn(f'Fixed-size fit rejected: {reason}')
        return None

    _, _, (polygon, x0, y0) = detection
    width, height, thickness = dimensions
    image_h, image_w = depth.shape
    ph, pw = polygon.shape
    px, py = int(0.25 * pw) + 10, int(0.25 * ph) + 10
    rx0, ry0 = max(0, x0 - px), max(0, y0 - py)
    rx1, ry1 = min(image_w, x0 + pw + px), min(image_h, y0 + ph + py)
    roi = depth[ry0:ry1, rx0:rx1]
    pts, ok = backproject_roi(roi, rx0, ry0, k)

    seed = np.zeros(roi.shape, np.uint8)
    seed[y0 - ry0:y0 - ry0 + ph, x0 - rx0:x0 - rx0 + pw] = polygon
    seed_inner = cv2.erode(seed, np.ones((7, 7), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    interior = pts[seed_inner & ok]
    if len(interior) < 50:
        return fail(f'only {len(interior)} valid depth pixels inside the seed')
    interior = interior[np.linspace(0, len(interior) - 1, min(400, len(interior)), dtype=int)]
    plane = fit_surface_plane(interior, plane_tolerance * 2)
    if plane is None:
        return fail('seed is not one flat surface')
    _, normal, plane_center = plane

    distance = np.abs((pts - plane_center) @ normal)
    gate = (ok & (distance < plane_tolerance)).astype(np.uint8)
    gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats(gate)
    if count < 2:
        return fail('no pixels on the lid plane')
    overlap = np.bincount(labels[seed_inner], minlength=count)
    overlap[0] = 0
    label = int(np.argmax(overlap))
    if overlap[label] == 0:
        return fail('plane region does not overlap the seed')
    lid_points = pts[labels == label]
    if len(lid_points) < 300:
        return fail(f'lid plane region has only {len(lid_points)} points')

    # Top-down metric occupancy image in the lid plane.
    origin = lid_points.mean(axis=0)
    helper = np.array([0., 1., 0.]) if abs(normal[1]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(helper, normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    ab = np.column_stack([(lid_points - origin) @ e1, (lid_points - origin) @ e2])
    margin = 0.05
    low = ab.min(axis=0) - margin
    ij = np.floor((ab - low) / resolution).astype(int)
    shape = ij.max(axis=0) + 1 + int(margin / resolution)
    if shape[0] * shape[1] > 400 * 400:
        return fail('lid plane region is far larger than the box (leaking into background plane)')
    occ = np.zeros((shape[1], shape[0]), np.uint8)
    occ[ij[:, 1], ij[:, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)

    def correlate(deci):
        template = rect_template(width, height, deci % 1800, resolution)
        score = cv2.filter2D(occ, -1, template, borderType=cv2.BORDER_CONSTANT)
        _, top, _, location = cv2.minMaxLoc(score)
        return top, location, template

    best = (-np.inf, 0, (0, 0), None)
    for deci in range(0, 1800, 30):          # 3 degree coarse sweep; rectangle has 180-degree symmetry
        top, location, template = correlate(deci)
        if top > best[0]:
            best = (top, deci, location, template)
    for deci in range(best[1] - 30, best[1] + 31, 5):   # 0.5 degree refinement
        top, location, template = correlate(deci)
        if top > best[0]:
            best = (top, deci % 1800, location, template)
    top, deci, (cx, cy), template = best
    coverage = top / max(1, np.count_nonzero(template > 0))
    if coverage < min_coverage:
        return fail(f'template coverage {coverage:.2f} below {min_coverage:.2f}')

    theta = math.radians(deci / 10.)
    ex = math.cos(theta) * e1 + math.sin(theta) * e2
    if ex[0] < 0:                            # stable convention for the 180-degree ambiguity
        ex = -ex
    ey = np.cross(normal, ex)
    center_face = (origin + (low[0] + (cx + .5) * resolution) * e1
                   + (low[1] + (cy + .5) * resolution) * e2)
    center_face -= ((center_face - plane_center) @ normal) * normal   # snap onto the lid plane
    rotation = np.column_stack([ex, ey, normal])
    return make_result(center_face + normal * thickness / 2, rotation, dimensions,
                       k, distortion, score=1. - coverage)


# ----------------------------------------------------------------------------
# Step 3: PnP fallback on the 2D corners
# ----------------------------------------------------------------------------
def estimate_target(depth, k, distortion, dimensions, detection,
                    depth_tolerance=0.04, reprojection_limit=4., warn=None):
    """PnP pose from recovered 2D corners; validated against measured depth."""
    def reject(reason):
        if warn is not None:
            warn(f'Pose rejected: {reason}')
        return None

    if detection is None:
        return reject('no brown rectangular face detected')
    if isinstance(detection, FaceDetection) and detection.pose_corners is None:
        return reject('2D brown detection available, but physical outer corners are unresolved')
    corners, pixel, (polygon, x0, y0) = detection
    depth_roi = depth[y0:y0 + polygon.shape[0], x0:x0 + polygon.shape[1]]
    u, v = np.rint(pixel).astype(int)
    u, v = u - x0, v - y0
    patch = depth_roi[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    mask = polygon[max(0, v - 3):v + 4, max(0, u - 3):u + 4] > 0
    valid = mask & np.isfinite(patch) & (patch > 0.1) & (patch < 10.)
    if np.count_nonzero(valid) < 5:
        return reject(f'center depth patch has {np.count_nonzero(valid)} valid pixels; '
                      'need >=5 finite depths between 0.1 and 10 m')
    z = float(np.median(patch[valid]))
    surface = deproject(pixel, z, k, distortion)
    width, height, thickness = dimensions
    object_points = np.array([[-width / 2, -height / 2, 0], [width / 2, -height / 2, 0],
                              [width / 2, height / 2, 0], [-width / 2, height / 2, 0]], np.float64)
    eroded = cv2.erode(polygon, np.ones((7, 7), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    grid_x = np.linspace(0, polygon.shape[1] - 1, min(32, polygon.shape[1]), dtype=int)
    grid_y = np.linspace(0, polygon.shape[0] - 1, min(32, polygon.shape[0]), dtype=int)
    xx, yy = np.meshgrid(grid_x, grid_y)
    candidate_z = depth_roi[yy, xx]
    valid = (eroded[yy, xx] > 0) & np.isfinite(candidate_z) & (candidate_z > 0.1) & (candidate_z < 10.)
    xx, yy = xx[valid], yy[valid]
    if len(xx) < 20:
        return reject(f'interior depth grid has {len(xx)} valid samples; need >=20 '
                      f'(checked {candidate_z.size} grid locations)')
    indices = np.linspace(0, len(xx) - 1, min(150, len(xx)), dtype=int)
    samples_uv = np.column_stack([xx[indices] + x0, yy[indices] + y0]).astype(np.float64)
    rays = cv2.undistortPoints(samples_uv.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    sample_z = depth_roi[yy[indices], xx[indices]]
    samples = np.column_stack([rays * sample_z[:, None], sample_z])
    plane = fit_surface_plane(samples, depth_tolerance)
    if plane is None:
        return reject('depth does not support one dominant flat face; use the closed patterned side')
    samples, measured_normal, _ = plane
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
                rejections.append(f'{label}: invalid face direction / behind camera')
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
            plane_error = float(np.median(np.abs((samples - translation) @ normal)))
            if np.dot(normal, measured_normal) < math.cos(math.radians(20)):
                rejections.append(f'{label}: pose normal disagrees with measured depth plane')
                continue
            if error > reprojection_limit or plane_error > depth_tolerance:
                rejections.append(f'{label}: reprojection={error:.2f} px (limit {reprojection_limit:.2f}), '
                                  f'plane depth error={plane_error:.4f} m (limit {depth_tolerance:.4f})')
                continue
            center_error = float(abs(np.dot(surface - translation, normal)))
            if center_error > depth_tolerance:
                rejections.append(f'{label}: center depth error={center_error:.4f} m')
                continue
            score = error + 100 * plane_error
            if best is None or score < best[0] - 1e-6:
                best = (score, translation + normal * thickness / 2, rotation)
    if best is None:
        return reject('no valid PnP pose; ' + '; '.join(rejections))
    normal = best[2][:, 2]
    # Range from measured depth, not from the (possibly slightly shrunken) corners.
    face_center = best[1] - normal * thickness / 2
    depth_correction = normal * np.median((samples - face_center) @ normal)
    center = best[1] + depth_correction
    return make_result(center, best[2], dimensions, k, distortion, score=best[0])


# ----------------------------------------------------------------------------
# Step 4: temporal filter
# ----------------------------------------------------------------------------
class PoseFilter:
    """EMA smoothing with jump rejection and 180-degree symmetry disambiguation."""
    def __init__(self, alpha=0.4, max_jump=0.05, max_angle_deg=15., max_age=1.0, max_rejects=3):
        self.alpha, self.max_jump = alpha, max_jump
        self.max_angle = math.radians(max_angle_deg)
        self.max_age, self.max_rejects = max_age, max_rejects
        self.reset()

    def reset(self):
        self.center = self.rotation = self.stamp = None
        self.rejects = 0

    @staticmethod
    def angle(a, b):
        return math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))

    def update(self, center, rotation, stamp):
        """Return the smoothed (center, rotation), or None if this frame is a rejected jump."""
        center, rotation = np.asarray(center, np.float64), np.asarray(rotation, np.float64)
        if self.center is not None and stamp - self.stamp > self.max_age:
            self.reset()
        if self.center is None:
            self.center, self.rotation, self.stamp = center, rotation, stamp
            return center, rotation
        # Symmetric face: use the 180-degree alternative closest to the previous pose.
        flipped = rotation @ np.diag([-1., -1., 1.])
        angle, rotation = min(((self.angle(self.rotation, r), r) for r in (rotation, flipped)),
                              key=lambda item: item[0])
        if np.linalg.norm(center - self.center) > self.max_jump or angle > self.max_angle:
            self.rejects += 1
            if self.rejects < self.max_rejects:
                return None
            self.center, self.rotation, self.stamp, self.rejects = center, rotation, stamp, 0
            return center, rotation
        self.rejects = 0
        a = self.alpha
        blended = (1 - a) * self.rotation + a * rotation
        u, _, vt = np.linalg.svd(blended)
        smooth = u @ np.diag([1., 1., np.sign(np.linalg.det(u @ vt))]) @ vt
        self.center = (1 - a) * self.center + a * center
        self.rotation, self.stamp = smooth, stamp
        return self.center, self.rotation


# ----------------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------------
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
            'brown_hsv_low': [0, 50, 20], 'brown_hsv_high': [40, 255, 255],
            'min_contour_area': 1200.0,
            'processing_rate_hz': 3.0,
            'capture_next_frame': False, 'capture_directory': '~/robot_ws/perception_captures',
            'depth_tolerance': 0.04, 'max_reprojection_error': 4.0,
            'use_depth_gate': True,        # restrict dark-border growth to the lid plane
            'lid_plane_tolerance': 0.02,   # m; raise to 0.03-0.04 for a warped lid
            'min_fit_coverage': 0.6,       # template coverage needed to accept the fixed-size fit
            'max_candidates': 3,           # brown seeds tried per frame
            'smoothing_alpha': 0.4}        # 1.0 disables smoothing
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
        for name in ('min_contour_area', 'depth_tolerance', 'max_reprojection_error', 'lid_plane_tolerance'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 < p['min_fit_coverage'] <= 1:
            raise ValueError('min_fit_coverage must be in (0, 1]')
        if not 0 < p['smoothing_alpha'] <= 1:
            raise ValueError('smoothing_alpha must be in (0, 1]')
        if int(p['max_candidates']) < 1:
            raise ValueError('max_candidates must be >= 1')
        rate = p['processing_rate_hz']
        if not math.isfinite(rate) or not 0 < rate <= 15:
            raise ValueError('processing_rate_hz must be positive and <= 15')
        cv2.setNumThreads(1)  # Avoid competing with camera encoding across CPU cores.
        self.bridge = CvBridge()
        self.pose_filter = PoseFilter(alpha=p['smoothing_alpha'])
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

    def intrinsics(self, info, rgb, bgr):
        """(K, D) for the color camera, (None, None) while CameraInfo is missing/mismatched."""
        if info is None or info.header.frame_id != rgb.header.frame_id:
            return None, None
        if (info.width, info.height) != (bgr.shape[1], bgr.shape[0]):
            raise ValueError('RGB and CameraInfo dimensions must match')
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError('Unsupported distortion model; use calibrated plumb_bob/rational_polynomial RGB')
        k = np.asarray(info.k, np.float64).reshape(3, 3)
        distortion = np.asarray(info.d, np.float64)
        if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.isfinite(distortion).all():
            raise ValueError('Invalid camera intrinsics')
        return k, distortion

    def matched_depth(self, rgb, bgr):
        """Depth in meters if it is recent enough and consistent with the RGB frame, else None."""
        msg = self.latest_depth
        if msg is None or abs(self.stamp_seconds(msg.header) - self.stamp_seconds(rgb.header)) > 0.1:
            return None
        if not rgb.header.frame_id or msg.header.frame_id != rgb.header.frame_id:
            raise ValueError('RGB and aligned depth must share the color optical frame')
        if (msg.width, msg.height) != (bgr.shape[1], bgr.shape[0]):
            raise ValueError('RGB and aligned depth dimensions must match')
        return depth_in_meters(self.bridge.imgmsg_to_cv2(msg, 'passthrough'), msg.encoding)

    def process_latest(self):
        rgb, self.latest_rgb = self.latest_rgb, None
        if rgb is None:
            return
        try:
            p = self.values
            bgr = self.bridge.compressed_imgmsg_to_cv2(rgb, 'bgr8')
            info = self.camera_info
            k, distortion = self.intrinsics(info, rgb, bgr)
            depth = self.matched_depth(rgb, bgr) if k is not None else None
            candidates = list(detect_faces(
                bgr, p['brown_hsv_low'], p['brown_hsv_high'], p['min_contour_area'],
                depth=depth if p['use_depth_gate'] else None, k=k,
                limit=int(p['max_candidates'])))
            if self.get_parameter('capture_next_frame').value:
                self.capture_frame(rgb, bgr, info, candidates[0] if candidates else None)
            if not candidates:
                self.marker_pub.publish(MarkerArray(markers=[Marker(
                    ns='detection_rays', id=4, action=Marker.DELETE)]))
                return
            if k is None:
                self.warn('Brown region detected; waiting for matching color CameraInfo')
                return
            # Viewing rays encode the 2D detection without claiming a 3D location.
            pixels = np.vstack([candidates[0][0], candidates[0][1]])
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
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
            if depth is None:
                self.warn('2D detection active; aligned depth is missing or older than 0.1 s')
                return

            reasons = []
            result = None
            for candidate in candidates:  # try every seed; do not stop at the first failure
                result = fit_fixed_rect(candidate, depth, k, distortion, self.dimensions,
                                        p['lid_plane_tolerance'], p['min_fit_coverage'],
                                        warn=reasons.append)
                if result is None and candidate.pose_corners is not None:
                    result = estimate_target(depth, k, distortion, self.dimensions, candidate,
                                             p['depth_tolerance'], p['max_reprojection_error'],
                                             warn=reasons.append)
                if result is not None:
                    break
            if result is None:
                self.warn('No depth-validated lid: ' + (reasons[0] if reasons else 'unknown'))
                return  # RViz markers expire after 0.5 s; no stale pose is republished.

            smoothed = self.pose_filter.update(result['center'], result['rotation'],
                                               self.stamp_seconds(rgb.header))
            if smoothed is None:
                self.warn('Pose jumped; holding previous pose until it persists')
                return
            result = make_result(smoothed[0], smoothed[1], self.dimensions, k, distortion, result['score'])
            if result is None:
                return
            self.publish_result(rgb, bgr, result)
        except (ValueError, CvBridgeError, cv2.error) as error:
            self.warn(str(error))

    def capture_frame(self, rgb, bgr, info, detection):
        """Save one camera sample locally, with no extra publisher or image stream."""
        self.set_parameters([Parameter('capture_next_frame', value=False)])
        directory = Path(self.get_parameter('capture_directory').value).expanduser()
        folder = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f') + 'Z')
        try:
            folder.mkdir(parents=True, exist_ok=False)
            suffix = '.png' if 'png' in rgb.format.lower() else '.jpg'
            (folder / ('rgb_original' + suffix)).write_bytes(bytes(rgb.data))
            cv2.imwrite(str(folder / 'rgb.png'), bgr)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(self.values['brown_hsv_low'], np.uint8),
                               np.array(self.values['brown_hsv_high'], np.uint8))
            cv2.imwrite(str(folder / 'hsv_mask.png'), mask)
            annotated = bgr.copy()
            if detection is not None:
                cv2.polylines(annotated, [detection[0].astype(np.int32)], True, (0, 255, 0), 2)
            cv2.imwrite(str(folder / 'detection.png'), annotated)
            data = {'rgb_frame': rgb.header.frame_id, 'rgb_stamp': self.stamp_seconds(rgb.header),
                    'hsv_low': list(self.values['brown_hsv_low']), 'hsv_high': list(self.values['brown_hsv_high'])}
            if info is not None:
                data.update({'K': list(info.k), 'D': list(info.d), 'distortion_model': info.distortion_model,
                             'width': info.width, 'height': info.height, 'info_frame': info.header.frame_id})
            depth = self.latest_depth
            if depth is not None:
                np.save(folder / 'depth_m.npy', depth_in_meters(self.bridge.imgmsg_to_cv2(depth, 'passthrough'), depth.encoding))
                age = abs(self.stamp_seconds(depth.header) - self.stamp_seconds(rgb.header))
                data.update({'depth_frame': depth.header.frame_id, 'depth_stamp': self.stamp_seconds(depth.header),
                             'depth_rgb_delta_s': age, 'depth_time_matched': age <= 0.1})
            (folder / 'camera_info.json').write_text(json.dumps(data, indent=2))
            self.get_logger().info(f'Saved D435i capture: {folder}')
        except (OSError, ValueError, CvBridgeError, cv2.error) as error:
            self.warn(f'Capture failed: {error}')

    @staticmethod
    def stamp_seconds(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def warn(self, message):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_warning >= 2:
            self.get_logger().warning(message)
            self.last_warning = now

    def publish_result(self, rgb, bgr, result):
        point = PointStamped(header=rgb.header)
        point.point.x, point.point.y, point.point.z = map(float, result['surface'])
        self.point_camera_pub.publish(point)
        if self.debug_pub.get_subscription_count():
            cv2.polylines(bgr, [np.rint(result['corners']).astype(np.int32)], True, (0, 255, 0), 2)
            u, v = np.rint(result['pixel']).astype(int)
            cv2.circle(bgr, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(bgr, f'center ({u}, {v}) px; depth {result["surface"][2]:.2f}m',
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            debug = self.bridge.cv2_to_compressed_imgmsg(bgr)
            debug.header = rgb.header
            self.debug_pub.publish(debug)
        try:
            transform = self.buffer.lookup_transform(self.values['base_frame'], rgb.header.frame_id,
                                                     Time.from_msg(rgb.header.stamp))
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