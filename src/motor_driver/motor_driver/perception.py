"""Brown/black rectangular-face RGB-D detection; independent of motor control.

Box axes: x along the 42 cm face edge, y along the 32 cm edge, z into the
5.5 cm thickness. Pose origin is the box center. Broad, side and end faces
share those axes. Unmarked faces cannot identify front/back or resolve every
180-degree symmetry; orientation is a consistent geometric estimate.
"""
import math
import heapq
import json
from pathlib import Path
from datetime import datetime, timezone
from rclpy.parameter import Parameter

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TransformStamped, Pose, Quaternion, Vector3
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA
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


def pixel_rays(pixels, k, distortion):
    """Convert image coordinates to optical-frame rays with z=1."""
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, k, distortion).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


def project_points(points, k, distortion):
    return cv2.projectPoints(np.asarray(points), np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)


def intersect_plane(pixels, normal, center, k, distortion):
    """Intersect forward camera rays with a measured face plane."""
    rays = pixel_rays(pixels, k, distortion)
    denominator = rays @ normal
    if np.any(denominator <= 1e-6):
        return None
    return rays * ((center @ normal) / denominator)[:, None]


def face_rotation(right, normal):
    right = right / np.linalg.norm(right)
    down = np.cross(normal, right)
    down /= np.linalg.norm(down)
    return np.column_stack([right, down, normal])


def deproject(pixel, z, k, distortion):
    return pixel_rays(pixel, k, distortion)[0] * z


class FaceDetection(tuple):
    """Image box is always usable; pose corners require a recovered boundary."""
    def __new__(cls, corners, pixel, region, pose_corners):
        result = super().__new__(cls, (corners, pixel, region))
        result.pose_corners = pose_corners
        return result


def ordered_corners(corners):
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    center = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1]-center[1], corners[:, 0]-center[0]))]
    return np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)


def fit_edge(points, values, transverse, minimum_span):
    """Fit the strongest edge band, ignoring isolated gradients and short clutter."""
    if len(values) < 5:
        return None
    bins = np.floor(values).astype(int)
    origin = bins.min()
    histogram = np.bincount(bins-origin)
    peak = np.argmax(np.convolve(histogram, np.ones(5), mode='same')) + origin
    selected = points[np.abs(values-peak) < 2]
    if len(selected) < 5 or np.ptp(selected @ transverse) < minimum_span:
        return None
    vx, vy, x, y = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    return np.array([vx, vy]), np.array([x, y])


def fit_outer_edges(contour, seed, offset):
    box=cv2.boxPoints(cv2.minAreaRect(seed)).astype(float)
    center=box.mean(0)-offset
    u=box[1]-box[0];w=np.linalg.norm(u);u/=w
    v=box[2]-box[1];h=np.linalg.norm(v);v/=h
    axes=np.column_stack([u,v]);pts=contour.reshape(-1,2).astype(float);coords=(pts-center)@axes
    lines=[]
    for axis,sign in [(0,-1),(1,-1),(0,1),(1,1)]:
        other=1-axis;extent=[w,h][other]
        band=(np.abs(coords[:,other])<extent*.35) & (coords[:,axis]*sign > [w,h][axis]*.48)
        line = fit_edge(pts[band], coords[band,axis], axes[:,other], extent*.3)
        if line is None:
            return None
        lines.append(line)
    corners=[]
    for a,b in [(0,1),(1,2),(2,3),(3,0)]:
        da,pa=lines[a];db,pb=lines[b];mat=np.column_stack([da,-db])
        if abs(np.linalg.det(mat))<.1:return
        corner=pa+np.linalg.solve(mat,pb-pa)[0]*da;corners.append(corner+offset)
    return np.array(corners)



def foreground_depth_mask(brown, depth):
    """Keep the dominant brown-depth cluster, never the similarly colored far background."""
    z=cv2.resize(depth,(brown.shape[1],brown.shape[0]),interpolation=cv2.INTER_NEAREST)
    valid=np.isfinite(z)&(z>0.1)&(z<8.)
    values=z[(brown>0)&valid]
    if len(values)<30:
        return None
    hist,edges=np.histogram(values,bins=np.arange(0,8.05,.05))
    peak=int(hist.argmax())
    near=(edges[peak]+edges[peak+1])/2
    band=valid&(np.abs(z-near)<max(.18,near*.12))
    # A plane is linear in inverse depth. Refine the loose range band using
    # only a small, deterministic sample, so nearby separate objects are removed
    # while a slanted face can retain its legitimate depth gradient.
    yy,xx=np.nonzero((brown>0)&band)
    if len(xx)>=30:
        ids=np.linspace(0,len(xx)-1,min(150,len(xx)),dtype=int)
        x=xx[ids]/brown.shape[1];y=yy[ids]/brown.shape[0]
        design=np.column_stack([x,y,np.ones(len(ids))]);sample=z[yy[ids],xx[ids]]
        rng=np.random.default_rng(7);best=None;count=0
        for _ in range(32):
            chosen=rng.choice(len(sample),3,replace=False)
            if abs(np.linalg.det(design[chosen]))<1e-6:continue
            model=np.linalg.solve(design[chosen],1/sample[chosen])
            inverse=design@model
            predicted=np.divide(1.,inverse,out=np.full_like(inverse,np.inf),where=inverse>0)
            inliers=np.abs(predicted-sample)<.025
            if inliers.sum()>count:best=inliers;count=int(inliers.sum())
        if best is not None and count>=max(25,int(.75*len(sample))):
            model=np.linalg.lstsq(design[best],1/sample[best],rcond=None)[0]
            gy,gx=np.indices(z.shape)
            inverse=model[0]*gx/brown.shape[1]+model[1]*gy/brown.shape[0]+model[2]
            predicted=np.divide(1.,inverse,out=np.full_like(inverse,np.inf),where=inverse>0)
            band=valid&(np.abs(z-predicted)<.035)&(np.abs(z-near)<.55)
    # One pixel at the reduced resolution covers RGB/depth edge registration.
    # Select this band globally before applying any tracking crop.
    return cv2.dilate(band.astype(np.uint8),np.ones((3,3),np.uint8))>0



def three_visible_edges(points, seed, offset):
    """Top and two sides for the bottom-clipped case, fitted from actual gradients."""
    x,y,w,h=cv2.boundingRect(seed)
    center=np.array([x+w/2,y+h/2])-offset
    pts=points.reshape(-1,2).astype(float)
    coords=pts-center
    lines=[]
    for axis,sign in [(0,-1),(1,-1),(0,1)]:
        other=1-axis
        band=(np.abs(coords[:,other])<[w,h][other]*.35)&(coords[:,axis]*sign>[w,h][axis]*.42)
        line = fit_edge(pts[band], coords[band,axis], np.eye(2)[other], [w,h][other]*.3)
        if line is None:
            return None
        lines.append(line)
    corners=[]
    for a,b in [(0,1),(1,2)]:
        da,pa=lines[a];db,pb=lines[b];matrix=np.column_stack([da,-db])
        if abs(np.linalg.det(matrix))<.2:return None
        corners.append(pa+np.linalg.solve(matrix,pb-pa)[0]*da+offset)
    return np.array(corners),lines


def interior_support(corners, brown, dark, foreground=None):
    """Score 32x24 interior samples; tolerate a small occlusion, not a large gap.

    Brown and nearby dark markings are appearance support. When available,
    the depth-selected surface is required as well. Image-clipped samples
    are unobserved rather than background.
    """
    corners=np.asarray(corners,np.float32)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):return None
    transform=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),corners)
    gx,gy=np.meshgrid(np.linspace(.06,.94,32),np.linspace(.06,.94,24))
    uv=cv2.perspectiveTransform(np.stack([gx,gy],axis=-1).astype(np.float32),transform)
    h,w=brown.shape
    visible=(uv[:,:,0]>=0)&(uv[:,:,0]<=w-1)&(uv[:,:,1]>=0)&(uv[:,:,1]<=h-1)
    if visible.sum()<96:return None
    x=np.clip(np.rint(uv[:,:,0]).astype(int),0,w-1)
    y=np.clip(np.rint(uv[:,:,1]).astype(int),0,h-1)
    brown_samples=brown[y,x]>0
    matches=brown_samples|(dark[y,x]>0)
    if foreground is not None:matches &= foreground[y,x]
    coverage=float(matches[visible].mean())
    brown_fraction=float(brown_samples[visible].mean())
    bad=(visible&~matches).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(bad,8)
    largest=float(stats[1:,cv2.CC_STAT_AREA].max()/visible.sum()) if count>1 else 0.
    if coverage<.82 or brown_fraction<.18 or largest>.18:return None
    return dict(score=coverage-.8*largest,
                coverage=coverage,largest_gap=largest)


def detect_face(bgr, hsv_low, hsv_high, minimum_area, warn=None, depth=None, search_region=None):
    """HSV locates brown; recover surrounding dark borders in a small local ROI."""
    scale = min(1.0, 320.0 / bgr.shape[1])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    foreground=foreground_depth_mask(brown,depth) if depth is not None else None
    if search_region is not None:
        left,top,right,bottom=np.rint(np.asarray(search_region)*scale).astype(int)
        restricted=np.zeros_like(brown)
        restricted[top:bottom,left:right]=brown[top:bottom,left:right]
        brown=restricted
    if foreground is not None:
        brown[~foreground]=0
    mask = cv2.morphologyEx(brown, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    # Dark pixels must be close to brown evidence. An attached sleeve must
    # not turn into an unlimited extension of the black border.
    nearby=cv2.dilate(brown,np.ones((31,31),np.uint8))>0
    dark=((hsv[:,:,2]<105)&nearby).astype(np.uint8)*255
    if foreground is not None:dark[~foreground]=0
    best_detection=None;best_score=-float('inf')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ranked = ((cv2.contourArea(c), i) for i, c in enumerate(contours))
    for area, index in heapq.nlargest(10, ranked, key=lambda item: item[0]):
        if area < minimum_area * scale**2 or area > small.shape[0]*small.shape[1]*0.85:
            continue
        seed = contours[index]
        x, y, w, h = cv2.boundingRect(seed)
        pad = max(5, int(min(w,h)*0.25))
        x0, y0 = max(0,x-pad), max(0,y-pad)
        x1, y1 = min(small.shape[1],x+w+pad), min(small.shape[0],y+h+pad)
        # Include nearby dark frame, but not pale floor merely sharing brown hue.
        combined = ((brown[y0:y1,x0:x1] > 0) | (dark[y0:y1,x0:x1]>0)).astype(np.uint8)*255
        if foreground is not None:
            combined[~foreground[y0:y1,x0:x1]]=0
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        local, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seed_center = (x+w/2-x0, y+h/2-y0)
        candidates = []
        for contour in local:
            if cv2.pointPolygonTest(contour,seed_center,False) < 0:
                continue
            hull = cv2.convexHull(contour)
            if not 0.85*area <= cv2.contourArea(hull) <= 1.8*area:
                continue
            for epsilon in (0.015,0.025,0.04):
                quad = cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
                if len(quad)==4 and cv2.isContourConvex(quad):
                    global_quad=quad.reshape(4,2)+[x0,y0]
                    if (np.any(global_quad[:,0]<=1) or np.any(global_quad[:,1]<=1)
                            or np.any(global_quad[:,0]>=small.shape[1]-2) or np.any(global_quad[:,1]>=small.shape[0]-2)):
                        continue  # Image-clipped/background contours cannot supply four face corners.
                    support=interior_support(ordered_corners(global_quad),brown,dark,foreground)
                    if support is not None:
                        candidates.append((support['score'],global_quad))
                    break
        edges = cv2.Canny(cv2.cvtColor(small[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY), 40, 100)
        if foreground is not None:
            edges[~foreground[y0:y1,x0:x1]] = 0
        pose_corners = None
        if candidates:
            pose_corners = ordered_corners(max(candidates,key=lambda q:q[0])[1]/scale)
            corners = pose_corners
        else:
            edge_points = cv2.findNonZero(edges)
            fitted = fit_outer_edges(edge_points,seed,np.array([x0,y0])) if edge_points is not None else None
            if fitted is not None:
                candidate = ordered_corners(fitted/scale)
                if (np.isfinite(candidate).all() and cv2.isContourConvex(candidate.astype(np.float32))
                        and cv2.contourArea(candidate.astype(np.float32)) < area/scale**2*2.0):
                    pose_corners = candidate
            corners = pose_corners if pose_corners is not None else ordered_corners(cv2.boxPoints(cv2.minAreaRect(seed))/scale)
        support=interior_support(corners*scale,brown,dark,foreground)
        if support is None:continue
        homography = cv2.getPerspectiveTransform(
            np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),corners.astype(np.float32))
        pixel = cv2.perspectiveTransform(np.array([[[0.5,0.5]]],np.float32),homography)[0,0].astype(np.float64)
        integer = np.rint(corners).astype(np.int32)
        rx,ry,rw,rh = cv2.boundingRect(integer)
        rx,ry = max(0,rx),max(0,ry)
        right,bottom = min(bgr.shape[1],rx+rw),min(bgr.shape[0],ry+rh)
        polygon = np.zeros((bottom-ry,right-rx),np.uint8)
        cv2.fillConvexPoly(polygon,integer-[rx,ry],255)
        detection=FaceDetection(corners,pixel,(polygon,rx,ry),pose_corners)
        detection.top_edge=None
        # Do not mistake a clipped frame boundary for a physical box edge.
        if y+h>=small.shape[0]-2 and y>2 and x>2 and x+w<small.shape[1]-2:
            points=cv2.findNonZero(edges)
            partial=three_visible_edges(points,seed,np.array([x0,y0])) if points is not None else None
            if partial is not None:
                detection.top_edge=partial[0]/scale
                detection.side_lines=[(v,pt/scale+np.array([x0,y0])/scale) for v,pt in (partial[1][0],partial[1][2])]
                detection.pose_corners=None
        # Compare candidates across seeds rather than returning the largest
        # connected component. Area is deliberately absent from this score.
        if support['score']>best_score:
            best_score=support['score'];best_detection=detection
            detection.interior=support
            detection.appearance=(brown,dark,scale)
    return best_detection


def fit_surface_plane(samples, tolerance):
    """Bounded deterministic RANSAC, then SVD; reject mixed tray/floor surfaces."""
    samples = np.asarray(samples,np.float64)
    if len(samples)<20:
        return None
    rng = np.random.default_rng(5)
    best = np.zeros(len(samples),bool)
    threshold = min(0.015,tolerance/2)
    for _ in range(32):
        a,b,c = samples[rng.choice(len(samples),3,replace=False)]
        normal = np.cross(b-a,c-a)
        length = np.linalg.norm(normal)
        if length<1e-8:
            continue
        normal /= length
        inliers = np.abs((samples-a)@normal)<threshold
        if inliers.sum()>best.sum():
            best=inliers
    if best.sum()<max(20,int(np.ceil(0.75*len(samples)))):
        return None
    clean=samples[best]
    center=clean.mean(axis=0)
    _,_,vectors=np.linalg.svd(clean-center,full_matrices=False)
    normal=vectors[-1]
    if normal@center<0:
        normal=-normal
    return clean,normal,center


def surface_samples(depth, detection, k, distortion, grid=24, erosion=0, max_depth=8.):
    """Sample at most 150 interior depths; share the same bounded geometry work."""
    polygon, x0, y0 = detection[2]
    roi = depth[y0:y0+polygon.shape[0], x0:x0+polygon.shape[1]]
    if erosion:
        polygon = cv2.erode(polygon, np.ones((erosion, erosion), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
    xx, yy = np.meshgrid(np.linspace(0, roi.shape[1]-1, min(grid, roi.shape[1]), dtype=int),
                         np.linspace(0, roi.shape[0]-1, min(grid, roi.shape[0]), dtype=int))
    z = roi[yy, xx]
    valid = (polygon[yy, xx] > 0) & np.isfinite(z) & (z > .1) & (z < max_depth)
    xx, yy = xx[valid], yy[valid]
    if len(xx) < 20:
        return np.empty((0, 3))
    ids = np.linspace(0, len(xx)-1, min(150, len(xx)), dtype=int)
    xx, yy = xx[ids], yy[ids]
    return pixel_rays(np.column_stack([xx+x0, yy+y0]), k, distortion) * roi[yy, xx, None]


def estimate_bottom_clipped(depth,k,distortion,dimensions,detection,tolerance,warn=None):
    """Infer only the hidden bottom using a depth plane, measured top, and known size."""
    samples = surface_samples(depth, detection, k, distortion, erosion=5)
    plane = fit_surface_plane(samples, tolerance)
    if plane is None:return None
    _,normal,plane_center=plane
    top=detection.top_edge
    top_xyz = intersect_plane(top, normal, plane_center, k, distortion)
    if top_xyz is None:
        return None
    edge=top_xyz[1]-top_xyz[0];length=np.linalg.norm(edge)
    width,height,thickness=dimensions
    errors=np.abs(length/np.array([width,height])-1.)
    axis=int(errors.argmin())
    if errors[axis]>.10:return None
    right=edge/length;down=np.cross(normal,right);down/=np.linalg.norm(down)
    vertical=height if axis==0 else width
    xyz=np.array([top_xyz[0],top_xyz[1],top_xyz[1]+down*vertical,top_xyz[0]+down*vertical])
    if not np.isfinite(xyz).all() or np.any(xyz[:,2]<=0.1):
        return None
    corners=project_points(xyz, k, distortion)
    if not np.isfinite(corners).all() or np.min(corners[2:,1])<depth.shape[0]-3:
        return None  # The inferred edge must actually lie outside the visible image.
    # Both measured side lines must agree with the inferred lower corners.
    for corner,(direction,point) in zip(corners[[3,2]],detection.side_lines):
        delta=corner-point
        if abs(direction[0]*delta[1]-direction[1]*delta[0])>6.:
            return None
    surface=xyz.mean(axis=0)
    axis_x=right if axis==0 else down
    rotation = face_rotation(axis_x, normal)
    pixel=project_points(surface[None], k, distortion)[0]
    return {'corners_camera':xyz,'pixel':pixel,'surface':surface,
            'center':surface+normal*thickness/2,'rotation':rotation,'corners':corners,
            'pose_source':'depth_plane_known_size_bottom_clipped'}


def estimate_target(bgr, depth, k, distortion, dimensions, hsv_low, hsv_high,
                    minimum_area=1200., depth_tolerance=0.04, reprojection_limit=4., detection=None,
                    warn=None):
    """Estimate a pose; report rejection details through an optional warning callback."""
    def reject(reason):
        if warn is not None:
            warn(f'Pose rejected: {reason}')
        return None

    detection = detection if detection is not None else detect_face(bgr, hsv_low, hsv_high, minimum_area, warn=warn, depth=depth)
    if detection is None:
        return reject('no brown rectangular face detected')
    if getattr(detection,'top_edge',None) is not None:
        return estimate_bottom_clipped(depth,k,distortion,dimensions,detection,depth_tolerance,warn)
    if isinstance(detection, FaceDetection) and detection.pose_corners is None:
        return reject('2D brown detection available, but physical outer corners are unresolved')
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
    samples = surface_samples(depth, detection, k, distortion, grid=32, erosion=7, max_depth=10.)
    if len(samples) < 20:
        return reject('fewer than 20 valid interior depth samples')
    plane = fit_surface_plane(samples, depth_tolerance)
    if plane is None:
        return reject('depth does not support one dominant flat face; use the closed patterned side')
    samples, measured_normal, measured_center = plane
    best = None
    for shift in range(2):  # 180-degree alternatives are symmetry-equivalent.
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
            if np.dot(normal, measured_normal) < math.cos(math.radians(20)):
                continue
            if error > reprojection_limit or plane_error > depth_tolerance:
                continue
            center_error = float(abs(np.dot(surface-translation, normal)))
            if center_error > depth_tolerance:
                continue
            # Stable image-corner convention breaks indistinguishable symmetry ties.
            score = error + 100 * plane_error
            if best is None or score < best[0] - 1e-6:
                best = (score, translation + normal * thickness/2, rotation)
    if best is None:
        # RGB-D alternative: intersect true image corners with the measured plane.
        # Validate physical edge lengths and orthogonality, instead of weakening PnP.
        xyz = intersect_plane(corners, measured_normal, measured_center, k, distortion)
        if xyz is None:
            return reject('face-corner rays are inconsistent with depth plane')
        lengths=np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1)
        for shift in range(2):
            pts=np.roll(xyz,shift,axis=0)
            edge_x=((pts[1]-pts[0])+(pts[2]-pts[3]))/2
            edge_y=((pts[3]-pts[0])+(pts[2]-pts[1]))/2
            measured=np.array([np.linalg.norm(edge_x),np.linalg.norm(edge_y)])
            if np.any(np.abs(measured/[width,height]-1)>0.10):
                continue
            if abs(np.dot(edge_x,edge_y)/np.prod(measured))>0.15:
                continue
            rotation = face_rotation(edge_x, measured_normal)
            if rotation[:, 1] @ edge_y < 0:
                continue
            center=xyz.mean(axis=0)+measured_normal*thickness/2
            best=(0.,center,rotation)
            break
        if best is None:
            return reject('PnP failed and measured face edges do not match known dimensions '
                          f'(edges {np.round(lengths,3).tolist()} m)')
    normal = best[2][:, 2]
    # Refine range with measured depth after validating the known face dimensions.
    face_center = best[1] - normal * thickness/2
    center = best[1] + normal * np.median((samples-face_center) @ normal)
    corners_camera = intersect_plane(corners, normal, center-normal*thickness/2, k, distortion)
    if corners_camera is None or np.any(corners_camera[:, 2] <= 0):
        return reject('face corners are parallel to the plane or behind the camera')
    return {'corners_camera': corners_camera, 'pixel': pixel, 'surface': surface, 'center': center, 'rotation': best[2], 'corners': corners}


def face_model(rotation,dimensions,kind):
    """Visible-face axes expressed in the fixed physical box frame."""
    width,height,thickness=dimensions
    if kind=='side':
        basis=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
        return rotation@basis,np.array([thickness,height,width])
    if kind=='end':
        basis=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
        return rotation@basis,np.array([width,thickness,height])
    return rotation,np.asarray(dimensions)


def pose_geometry(center, rotation, dimensions, k, distortion, **metadata):
    """One fixed-size model supplies outline, center marker and reported pose."""
    face_kind=metadata.get('face_kind','broad')
    local_rotation,local_dimensions=face_model(rotation,dimensions,face_kind)
    width,height,thickness=local_dimensions
    face=center-local_rotation[:,2]*thickness/2
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
    xyz=face+signs[:,:1]*local_rotation[:,0]*width/2+signs[:,1:]*local_rotation[:,1]*height/2
    if not np.isfinite(xyz).all() or np.any(xyz[:,2]<=.05):return None
    pixels=project_points(np.vstack([xyz, face]), k, distortion)
    return dict(center=center,rotation=rotation,surface=face,corners_camera=xyz,
                corners=pixels[:4],pixel=pixels[4],**metadata)


def candidate_plane(depth,detection,k,distortion,tolerance=.04):
    cached=getattr(detection,'plane',None)
    if cached is not None:return cached
    return fit_surface_plane(surface_samples(depth, detection, k, distortion), tolerance)


def fit_visible_face(bgr,depth,k,distortion,dimensions,detection,measurement=None,previous=None):
    """Sparse fixed-size local search scored against depth AND visible RGB edges.

    Unlike the other branch's all-angle image correlations, project only a
    bounded set of model samples. Off-image samples are unobserved, not empty.
    """
    plane=candidate_plane(depth,detection,k,distortion)
    if plane is None:return None
    _,normal,plane_center=plane
    width,height,thickness=dimensions
    corners=detection[0]
    observed_xyz = intersect_plane(corners, normal, plane_center, k, distortion)
    if observed_xyz is None:
        return None
    edge=(observed_xyz[1]-observed_xyz[0]+observed_xyz[2]-observed_xyz[3])/2
    edge-=normal*(edge@normal)
    if np.linalg.norm(edge)<1e-6:return None
    ex=edge/np.linalg.norm(edge);ey=np.cross(normal,ex)
    seeds=[]
    if measurement is not None:
        seeds.append((measurement['surface'],measurement['rotation'][:,0],True))
    else:
        for axis in (ex,ey):
            down=np.cross(normal,ex)
            center=observed_xyz.mean(axis=0)
            # Use the measured top to initialize a bottom-clipped face.
            if getattr(detection,'top_edge',None) is not None:
                top_xyz = intersect_plane(detection.top_edge, normal, plane_center, k, distortion)
                if top_xyz is None:
                    continue
                vertical=height if abs(axis@ex)>.9 else width
                center=top_xyz.mean(axis=0)+down*vertical/2
            seeds.append((center,axis,False))
    if previous is not None:
        seeds.append((previous['surface'],previous['rotation'][:,0],True))
    h,w=depth.shape
    allow_partial=bool(getattr(detection,'top_edge',None) is not None or
                       np.any(corners[:,0]<3) or np.any(corners[:,0]>w-4) or
                       np.any(corners[:,1]<3) or np.any(corners[:,1]>h-4))
    # Restrict edge computation to the candidate neighborhood.
    x0=max(0,int(corners[:,0].min())-30);x1=min(w,int(corners[:,0].max())+31)
    y0=max(0,int(corners[:,1].min())-30);y1=min(h,int(corners[:,1].max())+31)
    if x1<=x0 or y1<=y0:return None
    gray=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    scale=min(1.,320./gray.shape[1])
    if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    edges=cv2.Canny(gray,40,100)
    distances=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    # 35 face-interior samples and 64 edge samples, independent of image area.
    gx,gy=np.meshgrid(np.linspace(-.42,.42,7)*width,np.linspace(-.42,.42,5)*height)
    interior=np.column_stack([gx.ravel(),gy.ravel()])
    unit=np.linspace(-.5,.5,16)
    border=np.vstack([np.column_stack([unit*width,np.full(16,-height/2)]),
                      np.column_stack([np.full(16,width/2),unit*height]),
                      np.column_stack([unit*width,np.full(16,height/2)]),
                      np.column_stack([np.full(16,-width/2),unit*height])])
    local=np.vstack([interior,border])
    centers=[];rotations=[];priors=[]
    for center,axis,anchored in seeds:
        axis=axis-normal*(axis@normal);axis/=np.linalg.norm(axis)
        other=np.cross(normal,axis)
        center=center-normal*((center-plane_center)@normal)
        angles=(-3.,0.,3.) if anchored else (-12.,-6.,0.,6.,12.)
        radius=.015 if anchored else .03
        for degrees in angles:
            angle=math.radians(degrees)
            rx=math.cos(angle)*axis+math.sin(angle)*other;ry=np.cross(normal,rx)
            for dx in (-radius,0.,radius):
                for dy in (-radius,0.,radius):
                    centers.append(center+rx*dx+ry*dy)
                    rotations.append(np.column_stack([rx,ry,normal]))
                    priors.append((dx/radius)**2+(dy/radius)**2+(degrees/12.)**2)
    centers=np.asarray(centers);rotations=np.asarray(rotations)
    xyz=centers[:,None,:]+local[None,:,0,None]*rotations[:,None,:,0]+local[None,:,1,None]*rotations[:,None,:,1]
    pixels=project_points(xyz.reshape(-1, 3), k, distortion).reshape(len(centers), -1, 2)
    # Reuse the configured appearance masks from detection.
    appearance=getattr(detection,'appearance',None)
    best=None
    for i,(points,uv) in enumerate(zip(xyz,pixels)):
        if np.any(points[:,2]<=.1) or not np.isfinite(uv).all():continue
        visible=(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
        inside=visible[:len(interior)]
        if inside.sum()<12 or (not allow_partial and inside.mean()<.98):continue
        loc=np.rint(uv[:len(interior)][inside]).astype(int)
        z=depth[loc[:,1],loc[:,0]]
        residual=np.abs(z-points[:len(interior)][inside,2])
        good=np.isfinite(z)&(z>.1)&(residual<.025)
        coverage=good.mean()
        if coverage<.75:continue
        appearance_error=0.
        if appearance is not None:
            brown_mask,dark_mask,appearance_scale=appearance
            samples=np.rint(uv[:len(interior)][inside]*appearance_scale).astype(int)
            samples[:,0]=np.clip(samples[:,0],0,brown_mask.shape[1]-1)
            samples[:,1]=np.clip(samples[:,1],0,brown_mask.shape[0]-1)
            color_good=(brown_mask[samples[:,1],samples[:,0]]>0)|(dark_mask[samples[:,1],samples[:,0]]>0)
            if (color_good&good).mean()<.75:continue
            appearance_error=2*(1-color_good.mean())
        projected=(uv[len(interior):]-[x0,y0])*scale
        edge_visible=visible[len(interior):]&(projected[:,0]>=0)&(projected[:,0]<distances.shape[1]-1)&(projected[:,1]>=0)&(projected[:,1]<distances.shape[0]-1)
        if edge_visible.sum()<20:continue
        loc=np.rint(projected[edge_visible]).astype(int)
        edge_error=np.minimum(distances[loc[:,1],loc[:,0]]/scale,15).mean()
        if edge_error>7.:continue
        score=float(edge_error+8*(1-coverage)+.25*priors[i]+appearance_error)
        # Strong measured borders should anchor position, not merely plane occupancy.
        if measurement is not None:
            score+=float(np.linalg.norm(centers[i]-measurement['surface']))*60
        if best is None or score<best[0]:best=(score,i,coverage,float(edge_error))
    if best is None:return None
    score,i,coverage,edge_error=best
    source=measurement.get('pose_source','measured_edges') if measurement else 'fixed_face_local_fit'
    return pose_geometry(centers[i]+normal*thickness/2,rotations[i],dimensions,k,distortion,
                         score=score,coverage=coverage,edge_error=edge_error,pose_source=source)


def fixed_face_fit(bgr,depth,k,distortion,dimensions,detection,measurement=None,previous=None):
    """Try real cuboid faces; a narrow edge is not a malformed broad face.

    Plain brown faces use their silhouette and depth, without requiring stripes.
    Each returned rotation still describes the same physical box axes.
    """
    result=fit_visible_face(bgr,depth,k,distortion,dimensions,detection,measurement,previous
                           if previous is not None and previous.get('face_kind','broad')=='broad' else None)
    if result is not None:
        result['face_kind']='broad'
        return result
    best=None
    for kind in ('side','end'):
        basis,local_dimensions=face_model(np.eye(3),dimensions,kind)
        if getattr(detection,'pose_corners',None) is not None:
            plane=candidate_plane(depth,detection,k,distortion)
            if plane is None:continue
            _,normal,center=plane
            xyz = intersect_plane(detection[0], normal, center, k, distortion)
            if xyz is None:
                continue
            lengths=np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1)
            measured=np.sort([(lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2])
            if np.any(np.abs(measured/np.sort(local_dimensions[:2])-1)>.18):continue
        local=fit_visible_face(bgr,depth,k,distortion,local_dimensions,detection)
        if local is None:continue
        # Full silhouettes must agree in area as well as interior depth. This
        # prevents a small face from fitting somewhere inside a larger object.
        if getattr(detection,'top_edge',None) is None:
            observed=cv2.contourArea(detection[0].astype(np.float32))
            predicted=cv2.contourArea(local['corners'].astype(np.float32))
            if observed<=0 or not .85< predicted/observed <1.2:continue
        local['rotation']=local['rotation']@basis.T
        local['face_kind']=kind
        local['pose_source']='depth_plane_known_size_'+kind
        if best is None or local['score']<best['score']:best=local
    return best


class PoseTracker:
    """Time-aware smoothing and motion gates; reject inconsistent jumps."""
    def __init__(self):
        self.result=None;self.stamp=None;self.pending=None;self.failures=0

    def search_region(self,k,distortion,dimensions,stamp,shape):
        if self.failures>=2 or self.result is None or stamp-self.stamp>1.0 or stamp<=self.stamp:return None
        result=pose_geometry(self.result['center'],self.result['rotation'],dimensions,k,distortion,face_kind=self.result.get('face_kind','broad'))
        if result is None:return None
        corners=result['corners'];lo=corners.min(axis=0);hi=corners.max(axis=0)
        margin=np.maximum(35.,(hi-lo)*.4)
        return (max(0,int(lo[0]-margin[0])),max(0,int(lo[1]-margin[1])),
                min(shape[1],int(hi[0]+margin[0])+1),min(shape[0],int(hi[1]+margin[1])+1))

    def update(self,result,stamp,dimensions,k,distortion):
        self.failures=0
        if self.stamp is not None and stamp<=self.stamp:return None
        if self.result is None or stamp-self.stamp>1.0:
            self.result=result;self.stamp=stamp;self.pending=None;return result
        dt=stamp-self.stamp
        old=self.result
        rotation=result['rotation']
        kind=result.get('face_kind','broad')
        basis,_=face_model(np.eye(3),dimensions,kind)
        symmetry=basis@np.diag([-1.,-1.,1.])@basis.T
        alternatives=[rotation,rotation@symmetry]
        def angle(r):return math.acos(float(np.clip((np.trace(old['rotation'].T@r)-1)/2,-1,1)))
        rotation=min(alternatives,key=angle)
        if np.linalg.norm(result['center']-old['center'])>.04+dt*.8 or angle(rotation)>math.radians(15+90*dt):
            if (self.pending is None or np.linalg.norm(result['center']-self.pending['center'])>.04+dt*.8
                    or math.acos(float(np.clip((np.trace(self.pending['rotation'].T@rotation)-1)/2,-1,1)))>math.radians(20)):
                self.pending=result;return None
            self.result=result;self.stamp=stamp;self.pending=None;return result
        tau=.1 if result.get('edge_error',99)<3 else .2
        alpha=1-math.exp(-dt/tau)
        center=old['center']*(1-alpha)+result['center']*alpha
        mixed=old['rotation']*(1-alpha)+rotation*alpha
        u,_,vt=np.linalg.svd(mixed)
        rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        metadata={key:result[key] for key in ('score','coverage','edge_error','pose_source','face_kind') if key in result}
        smooth=pose_geometry(center,rotation,dimensions,k,distortion,**metadata)
        self.result=smooth;self.stamp=stamp;self.pending=None
        return smooth


def make_marker(header, namespace, identifier, kind, scale, color, *, pose=None, points=(), lifetime=.5):
    """Common RViz defaults; callers specify only the visual's actual contents."""
    marker = Marker(header=header, ns=namespace, id=identifier, type=kind, action=Marker.ADD,
                    pose=pose if pose is not None else Pose(orientation=Quaternion(w=1.)),
                    scale=Vector3(x=float(scale[0]), y=float(scale[1]), z=float(scale[2])),
                    color=ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3]), points=list(points))
    marker.lifetime.sec = int(lifetime)
    marker.lifetime.nanosec = int((lifetime-int(lifetime))*1e9)
    return marker


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
            'processing_rate_hz': 6.0,
            'capture_next_frame': False, 'capture_directory': '~/robot_ws/perception_captures',
            'depth_tolerance': 0.04, 'max_reprojection_error': 4.0}
        parameters = self.declare_parameters('', list(defaults.items()))
        self.values = {parameter.name: parameter.value for parameter in parameters}
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
        self.tracker = PoseTracker()
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
            depth_image=None
            candidate_depth=self.latest_depth
            if (candidate_depth is not None and candidate_depth.header.frame_id==rgb.header.frame_id
                    and abs(self.stamp_seconds(candidate_depth.header)-self.stamp_seconds(rgb.header))<=.1
                    and (candidate_depth.width,candidate_depth.height)==(bgr.shape[1],bgr.shape[0])):
                depth_image=depth_in_meters(self.bridge.imgmsg_to_cv2(candidate_depth,'passthrough'),candidate_depth.encoding)
            info = self.camera_info
            region=None
            if not hasattr(self,'tracker'):
                self.tracker=PoseTracker()
            if self.tracker.result is not None and info is not None and info.header.frame_id==rgb.header.frame_id:
                k=np.asarray(info.k,np.float64).reshape(3,3)
                if np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0:
                    region=self.tracker.search_region(k,np.asarray(info.d,np.float64),self.dimensions,
                                                     self.stamp_seconds(rgb.header),bgr.shape)
            detection = detect_face(bgr, self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                    self.values['min_contour_area'], warn=self.warn, depth=depth_image,
                                    search_region=region)
            if detection is None and region is not None:
                detection=detect_face(bgr,self.values['brown_hsv_low'],self.values['brown_hsv_high'],
                                      self.values['min_contour_area'],warn=self.warn,depth=depth_image)
            if self.get_parameter('capture_next_frame').value:
                self.capture_frame(rgb, bgr, info, detection)
            if detection is None:
                self.tracker.failures+=1
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
            points = [point for x, y in xy for point in (Point(), Point(x=float(x), y=float(y), z=1.))]
            rays = make_marker(rgb.header, 'detection_rays', 4, Marker.LINE_LIST,
                               (.002, 0., 0.), (0., 1., 0., .15), points=points, lifetime=.6)
            self.marker_pub.publish(MarkerArray(markers=[rays]))
            depth = self.latest_depth
            if depth is None or abs(self.stamp_seconds(depth.header)-self.stamp_seconds(rgb.header)) > 0.1:
                self.warn('2D detection active; aligned depth is missing or older than 0.1 s')
                return
            self.process(rgb, depth, info, bgr=bgr, detection=detection, depth_m=depth_image)
        except (ValueError, CvBridgeError, cv2.error) as error:
            self.warn(str(error))

    def capture_frame(self, rgb, bgr, info, detection):
        """Save one camera sample locally, with no extra publisher or image stream."""
        self.set_parameters([Parameter('capture_next_frame', value=False)])
        directory = Path(self.get_parameter('capture_directory').value).expanduser()
        folder = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'Z')
        try:
            folder.mkdir(parents=True,exist_ok=False)
            suffix = '.png' if 'png' in rgb.format.lower() else '.jpg'
            (folder/('rgb_original'+suffix)).write_bytes(bytes(rgb.data))
            cv2.imwrite(str(folder/'rgb.png'),bgr)
            hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,np.array(self.values['brown_hsv_low'],np.uint8),np.array(self.values['brown_hsv_high'],np.uint8))
            cv2.imwrite(str(folder/'hsv_mask.png'),mask)
            annotated=bgr.copy()
            if detection is not None:
                cv2.polylines(annotated,[detection[0].astype(np.int32)],True,(0,255,0),2)
            cv2.imwrite(str(folder/'detection.png'),annotated)
            data={'rgb_frame':rgb.header.frame_id,'rgb_stamp':self.stamp_seconds(rgb.header),
                  'hsv_low':list(self.values['brown_hsv_low']),'hsv_high':list(self.values['brown_hsv_high'])}
            if info is not None:
                data.update({'K':list(info.k),'D':list(info.d),'distortion_model':info.distortion_model,
                             'width':info.width,'height':info.height,'info_frame':info.header.frame_id})
            depth=self.latest_depth
            if depth is not None:
                np.save(folder/'depth_m.npy',depth_in_meters(self.bridge.imgmsg_to_cv2(depth,'passthrough'),depth.encoding))
                age=abs(self.stamp_seconds(depth.header)-self.stamp_seconds(rgb.header))
                data.update({'depth_frame':depth.header.frame_id,'depth_stamp':self.stamp_seconds(depth.header),
                             'depth_rgb_delta_s':age,'depth_time_matched':age<=0.1})
            (folder/'camera_info.json').write_text(json.dumps(data,indent=2))
            self.get_logger().info(f'Saved D435i capture: {folder}')
        except (OSError,ValueError,CvBridgeError,cv2.error) as error:
            self.warn(f'Capture failed: {error}')

    @staticmethod
    def stamp_seconds(header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def warn(self, message):
        now = self.get_clock().now().nanoseconds / 1e9
        if now-self.last_warning >= 2:
            self.get_logger().warning(message)
            self.last_warning = now

    def process(self, rgb, depth_msg, info, bgr=None, detection=None, depth_m=None):
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
            depth = depth_m if depth_m is not None else depth_in_meters(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'), depth_msg.encoding)
            if detection is None:
                detection = detect_face(bgr, self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                        self.values['min_contour_area'], warn=self.warn)
            if detection is None:
                self.warn('No brown rectangular target found; check lighting and brown_hsv_low/high')
                return
            # Keep measured perspective edges as the anchor; the fixed-size depth
            # fit can also recover a pose when four physical corners are unavailable.
            measurement = estimate_target(bgr, depth, k, distortion, self.dimensions,
                                     self.values['brown_hsv_low'], self.values['brown_hsv_high'],
                                     self.values['min_contour_area'], self.values['depth_tolerance'],
                                     self.values['max_reprojection_error'], detection=detection)
            if not hasattr(self,'tracker'):
                self.tracker=PoseTracker()
            stamp=Perception.stamp_seconds(rgb.header)
            previous=self.tracker.result if self.tracker.stamp is not None and 0<stamp-self.tracker.stamp<1 else None
            result=fixed_face_fit(bgr,depth,k,distortion,self.dimensions,detection,
                                  measurement=measurement,previous=previous)
            if result is None and measurement is not None:
                result=pose_geometry(measurement['center'],measurement['rotation'],self.dimensions,k,distortion,
                                     pose_source=measurement.get('pose_source','measured_edges'))
            appearance=getattr(detection,'appearance',None)
            if result is not None and appearance is not None:
                brown_mask,dark_mask,scale=appearance
                if interior_support(result['corners']*scale,brown_mask,dark_mask) is None:
                    result=None
            if result is None:
                self.tracker.failures+=1
                self.warn('Pose rejected: insufficient consistent face depth, appearance and visible boundary support')
                return
            result=self.tracker.update(result,stamp,self.dimensions,k,distortion)
            if result is None:
                return  # No stale pose is republished while a jump is being checked.
            point = PointStamped(header=rgb.header)
            point.point.x, point.point.y, point.point.z = map(float, result['surface'])
            self.point_camera_pub.publish(point)
            if self.debug_pub.get_subscription_count():
                cv2.polylines(bgr, [result['corners'].astype(np.int32)], True, (0, 255, 0), 2)
                u, v = np.rint(result['pixel']).astype(int)
                cv2.circle(bgr, (u, v), 4, (0, 255, 255), -1)
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
        contour_points = [Point(x=float(x), y=float(y), z=float(z))
                          for x, y, z in np.vstack([corners_camera, corners_camera[:1]])]
        center = Point(x=float(center_camera[0]), y=float(center_camera[1]), z=float(center_camera[2]))
        markers = [
            make_marker(point.header, 'detected_surface', 0, Marker.SPHERE,
                        (.035, .035, .035), (0., 1., 0., 1.),
                        pose=Pose(position=point.point, orientation=Quaternion(w=1.))),
            make_marker(pose.header, 'box_pose', 1, Marker.CUBE,
                        self.dimensions, (0., 0., 1., .35), pose=pose.pose),
            make_marker(camera_header, 'detected_contour', 2, Marker.LINE_STRIP,
                        (.004, 0., 0.), (0., 1., 0., 1.), points=contour_points),
            make_marker(camera_header, 'box_center_camera', 3, Marker.SPHERE,
                        (.01, .01, .01), (1., 1., 0., 1.),
                        pose=Pose(position=center, orientation=Quaternion(w=1.)))
        ]
        self.marker_pub.publish(MarkerArray(markers=markers))


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
