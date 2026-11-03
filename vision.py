"""RGB-only perception: known-color upright 30 mm cubes on a known plane.

No object poses, simulator segmentation IDs or depth buffers are read here.
Simulation supplies camera calibration and synchronized camera poses only.
"""
from dataclasses import dataclass

import cv2
import numpy as np


class VisionError(ValueError):
    pass


COLORS = {
    "red_cube": ((0, 12), (170, 179)),
    "blue_cube": ((100, 135),),
    "yellow_cube": ((18, 40),),
    "left_box": ((45, 90),),
    "right_box": ((136, 169),),
}


@dataclass
class Frame:
    rgb: np.ndarray
    stamp: float
    K: np.ndarray
    position: np.ndarray
    rotation: np.ndarray  # OpenCV camera axes: right, down, forward, in base frame.

    def pixel_on_plane(self, pixel, z):
        ray = self.rotation @ np.linalg.solve(self.K, [*pixel, 1.])
        if abs(ray[2]) < 1e-8:
            raise VisionError("PLANE_PARALLEL: image ray does not intersect the table")
        distance = (z - self.position[2]) / ray[2]
        if distance <= 0:
            raise VisionError("PLANE_BEHIND_CAMERA")
        return self.position + distance * ray

    def project(self, points):
        camera = (np.asarray(points) - self.position) @ self.rotation
        if np.any(camera[:, 2] <= 0):
            raise VisionError("TARGET_BEHIND_CAMERA")
        pixels = camera @ self.K.T
        return pixels[:, :2] / pixels[:, 2:3]


def color_mask(rgb, name):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLORS[name]:
        mask |= cv2.inRange(hsv, (lo, 100, 80), (hi, 255, 255))
    if name == "blue_cube":
        rgb16 = rgb.astype(np.int16)
        mask[(rgb16[:, :, 2] - rgb16[:, :, 0] < 60) | (rgb16[:, :, 2] - rgb16[:, :, 1] < 30)] = 0
    return mask


class VisionSystem:
    def __init__(self, model, cameras):
        self.model, self.cameras = model, cameras
        self.frames = {}
        self.objects, self.boxes = {}, {}
        self.box_pixels = {}
        self.stamp = -1.
        self.last_error = None
        self.sequence = 0

    def capture(self, data):
        # Called on the physics thread: RGB and camera FK have the same timestamp.
        images = self.cameras.capture(data)
        frames = {}
        for name, rgb in images.items():
            h, w = rgb.shape[:2]
            f = .5 * h / np.tan(np.deg2rad(self.model.cam_fovy[self.model.camera(name).id]) / 2)
            K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.]])
            camera = data.camera(name)
            frames[name] = Frame(rgb, float(data.time), K, camera.xpos.copy(),
                                 camera.xmat.reshape(3, 3).copy() @ np.diag([1., -1., -1.]))
        self.frames = frames
        self.sequence += 1
        return frames

    def require_fresh(self, now):
        if not self.frames or any(not 0 <= now - f.stamp <= .3 for f in self.frames.values()):
            raise VisionError("STALE_FRAME: RGB and camera pose must be current")

    def locate(self, frame, name, surface_z=0.):
        mask = color_mask(frame.rgb, name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        is_box = name.endswith("_box")
        z = .022 if is_box else surface_z + .030
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 16:
                continue
            rect = cv2.minAreaRect(contour)
            pixels = cv2.boxPoints(rect)
            world = np.array([frame.pixel_on_plane(p, z) for p in pixels])
            if is_box:
                edge = world[1, :2] - world[0, :2]
                if min(abs(edge)) / max(np.linalg.norm(edge), 1e-8) > .12:
                    continue  # This controller uses axis-aligned box slots.
            lengths = sorted(np.linalg.norm(world[(i + 1) % 4, :2] - world[i, :2]) for i in (0, 1))
            expected = (.07, .11) if is_box else (.03, .03)
            if any(abs(length - size) > (.014 if is_box else .009) for length, size in zip(lengths, expected)):
                continue
            if area / max(rect[1][0] * rect[1][1], 1.) < .72:
                continue
            center = frame.pixel_on_plane(rect[0], z)
            if not (.055 < center[0] < .34 and abs(center[1]) < .21):
                continue
            center[2] = .004 if is_box else surface_z + .015
            candidates.append({"xyz": center, "bbox": cv2.boundingRect(contour),
                               "pixels": pixels, "stamp": frame.stamp})
        if len(candidates) != 1:
            code = "NOT_VISIBLE" if not candidates else "AMBIGUOUS"
            raise VisionError(f"{code}: {name}, {len(candidates)} valid color/size candidates")
        return candidates[0]

    def scan(self, data):
        self.capture(data)
        self.require_fresh(data.time)
        frame = self.frames["overhead_cam"]
        # Publish only a complete observation; never retain stale objects on failure.
        self.objects, self.boxes = {}, {}
        self.stamp = -1.
        try:
            boxes = {n: self.locate(frame, n) for n in ("left_box", "right_box")}
            objects = {}
            for name in ("red_cube", "blue_cube", "yellow_cube"):
                detection = self.locate(frame, name)
                in_box = any(np.all(np.abs(detection["xyz"][:2] - b["xyz"][:2]) < [.052, .032])
                             for b in boxes.values())
                if in_box:
                    detection = self.locate(frame, name, .004)
                objects[name] = detection
            self.objects, self.boxes = objects, boxes
            self.stamp, self.last_error = float(data.time), None
        except VisionError as exc:
            self.last_error = str(exc)
            raise
        return objects, boxes

    def wrist_confirm(self, data, name, expected_xyz):
        """Verify the colored cube near the planned grasp using the wrist RGB.

        This is image-space confirmation, not an invented RGB depth estimate.
        A fully visible 3D box silhouette must overlap its predicted projection.
        """
        self.capture(data)
        self.require_fresh(data.time)
        f = self.frames["wrist_cam"]
        corners = np.array([[x, y, z] for x in (-.015, .015)
                            for y in (-.015, .015) for z in (-.015, .015)]) + expected_xyz
        pixels = f.project(corners)
        lo, hi = np.floor(pixels.min(0)).astype(int) - 8, np.ceil(pixels.max(0)).astype(int) + 8
        h, w = f.rgb.shape[:2]
        lo = np.maximum(lo, [0, 0]); hi = np.minimum(hi, [w, h])
        if np.any(hi <= lo):
            raise VisionError("WRIST_TARGET_OUT_OF_VIEW")
        mask = color_mask(f.rgb, name)
        roi = mask[lo[1]:hi[1], lo[0]:hi[0]]
        if cv2.countNonZero(roi) < max(30, .10 * roi.size):
            raise VisionError("WRIST_TARGET_MISSING: reobserve before descending")
        ys, xs = np.nonzero(roi)
        observed = np.array([np.mean(xs) + lo[0], np.mean(ys) + lo[1]])
        expected = (pixels.min(0) + pixels.max(0)) / 2
        if np.linalg.norm(observed - expected) > max(12., .20 * np.linalg.norm(hi - lo)):
            raise VisionError("WRIST_TARGET_MOVED: target differs from planned grasp")
        return {"camera": "wrist_cam", "stamp": f.stamp,
                "color_pixels": int(len(xs)), "pixel_error": round(float(np.linalg.norm(observed - expected)), 2)}

    def telemetry(self):
        return {"source": "rgb_plane_geometry", "stamp": self.stamp,
                "objects": {n: np.round(v["xyz"], 5).tolist() for n, v in self.objects.items()},
                "boxes": {n: np.round(v["xyz"], 5).tolist() for n, v in self.boxes.items()},
                "last_error": self.last_error}
