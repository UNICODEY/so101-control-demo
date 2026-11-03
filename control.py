"""SO-101 scene, inverse kinematics and feedback-driven pick-and-place control."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# Scene
HALF = .015
BOX_FLOOR_Z = .004
BOX_INNER = (.052, .032)
BOX_WALL_TOP = .022

# Ideal RGB cameras. Advertised FOV is treated as diagonal (not calibrated).
CAMERAS = {
    "wrist_cam": {"width": 640, "height": 480, "diagonal_fov": 120.},
    "overhead_cam": {"width": 1920, "height": 1080, "diagonal_fov": 160.},
}


def camera_fovy(spec):
    aspect = spec["width"] / spec["height"]
    return float(np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(spec["diagonal_fov"]) / 2)
                                         / np.sqrt(1 + aspect * aspect))))


def layout(shifted=False):
    objects = {
        "red_cube": {"xy": [.23, -.06], "rgba": ".9 .12 .08 1"},
        "blue_cube": {"xy": [.25, 0.], "rgba": ".12 .35 .95 1"},
        "yellow_cube": {"xy": [.23, .06], "rgba": ".95 .75 .08 1"},
    }
    boxes = {
        "left_box": {"xy": [.16, .12], "rgba": ".15 .65 .35 1"},
        "right_box": {"xy": [.16, -.12], "rgba": ".65 .25 .65 1"},
    }
    if shifted:
        for i, obj in enumerate(objects.values()):
            obj["xy"][0] -= .005
            obj["xy"][1] += .004 if i != 1 else -.004
        for box in boxes.values():
            box["xy"][0] += .005
    return objects, boxes


def build(model_path, shifted=False):
    directory = Path(model_path).resolve().parent
    root = ET.parse(directory / "so101.xml").getroot()
    root.find("compiler").set("meshdir", str(directory / "assets"))
    root.find("actuator/position[@name='gripper']").set("forcerange", "-.30 .30")
    root.find("visual/scale").set("framelength", ".03")
    asset, world = root.find("asset"), root.find("worldbody")
    # Reuse the upstream wrist camera and its physical mount; fixed world camera
    # looks along -Z, with robot +X right and +Y up in the image.
    wrist_camera = root.find(".//camera[@name='wrist_cam']")
    if wrist_camera is None:
        raise ValueError("The SO-101 model must include wrist_cam; use the pinned model")
    wrist_camera.set("fovy", str(camera_fovy(CAMERAS["wrist_cam"])))
    ET.SubElement(world, "camera", name="overhead_cam", mode="fixed",
                  pos=".18 0 .45", xyaxes="1 0 0 0 1 0",
                  fovy=str(camera_fovy(CAMERAS["overhead_cam"])))
    visual_global = root.find("visual/global")
    if visual_global is None:
        visual_global = ET.SubElement(root.find("visual"), "global")
    visual_global.set("offwidth", "1920")
    visual_global.set("offheight", "1080")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".18 .22 .28", rgb2=".25 .30 .36", width="256", height="256")
    ET.SubElement(asset, "material", name="ground", texture="grid", texrepeat="20 20", texuniform="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .05", material="ground")
    ET.SubElement(world, "light", pos=".3 -.4 1.5", dir="0 0 -1", directional="true")
    ET.SubElement(root.find(".//body[@name='gripper']"), "site", name="tcp", pos=".007 0 -.09", size=".002", rgba="0 .8 1 1", group="3")
    objects, boxes = layout(shifted)
    for name, spec in objects.items():
        x, y = spec["xy"]
        body = ET.SubElement(world, "body", name=name, pos=f"{x} {y} {HALF}")
        ET.SubElement(body, "freejoint", name=name + "_free")
        ET.SubElement(body, "geom", name=name + "_geom", type="box", size=".015 .015 .015", mass=".025", rgba=spec["rgba"], friction="1 .005 .0001", condim="6", solref=".01 1")
    for name, spec in boxes.items():
        x, y = spec["xy"]
        body = ET.SubElement(world, "body", name=name, pos=f"{x} {y} 0")
        common = dict(type="box", rgba=spec["rgba"], friction="1 .005 .0001")
        ET.SubElement(body, "geom", name=name + "_bottom", pos="0 0 .002", size=".055 .035 .002", **common)
        ET.SubElement(body, "geom", name=name + "_wall_xp", pos=".0535 0 .013", size=".0015 .035 .009", **common)
        ET.SubElement(body, "geom", name=name + "_wall_xm", pos="-.0535 0 .013", size=".0015 .035 .009", **common)
        ET.SubElement(body, "geom", name=name + "_wall_yp", pos="0 .0335 .013", size=".052 .0015 .009", **common)
        ET.SubElement(body, "geom", name=name + "_wall_ym", pos="0 -.0335 .013", size=".052 .0015 .009", **common)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, objects, boxes


def contained(data, object_id, box):
    """All eight cube corners must lie inside the box in x/y, near its floor."""
    body = data.body(object_id)
    rotation = body.xmat.reshape(3, 3)
    extent = np.abs(rotation) @ np.full(3, HALF)
    delta = np.abs(body.xpos[:2] - np.asarray(box["xy"]))
    return bool(np.all(delta + extent[:2] <= np.asarray(BOX_INNER) - .001)
                and abs(body.xpos[2] - (BOX_FLOOR_Z + HALF)) < .004)


# Inverse kinematics
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

class Robot:
    def __init__(self, model):
        self.m = model
        self.a = np.array([model.actuator(n).id for n in NAMES])
        self.j = np.array([model.joint(n).id for n in NAMES])
        self.q = model.jnt_qposadr[self.j]
        self.v = model.jnt_dofadr[self.j]
        self.limits = np.column_stack((np.maximum(model.jnt_range[self.j, 0], model.actuator_ctrlrange[self.a, 0]),
                                       np.minimum(model.jnt_range[self.j, 1], model.actuator_ctrlrange[self.a, 1])))
        if model.nu != 6 or not np.array_equal(model.actuator_trnid[self.a, 0], self.j):
            raise ValueError("Unexpected model actuator mapping")
        self.ik_data = mujoco.MjData(model)  # Never used as the live physics state.

    def tcp_at(self, q):
        self.ik_data.qpos[self.q] = q
        mujoco.mj_kinematics(self.m, self.ik_data)
        return self.ik_data.site("tcp").xpos.copy()

    def ik(self, target, seed=None):
        q = np.array(seed if seed is not None else [0., -.5, 1., 1., 0.], dtype=float)
        d = self.ik_data

        def residual(angles):
            d.qpos[self.q[:5]] = angles
            mujoco.mj_kinematics(self.m, d)
            rotation = d.body("gripper").xmat.reshape(3, 3)
            return np.r_[d.site("tcp").xpos - target,
                         .15 * (rotation[:, 2] - [0, 0, 1]),
                         .10 * (rotation[:, 0] - [1, 0, 0])]

        for _ in range(300):
            error = residual(q)
            if np.linalg.norm(error) < 1e-5:
                return q.copy()
            jacobian = np.column_stack([(residual(q + np.eye(5)[j] * 1e-5) - error) / 1e-5 for j in range(5)])
            delta = -np.linalg.solve(jacobian.T @ jacobian + np.eye(5) * 1e-5, jacobian.T @ error)
            delta *= min(1., .15 / max(float(np.max(np.abs(delta))), 1e-10))
            q = np.clip(q + delta, self.limits[:5, 0] + .001, self.limits[:5, 1] - .001)
        raise ValueError(f"IK target unreachable: {np.round(target, 4).tolist()}, residual={np.linalg.norm(residual(q)):.5f}")


# Pick-and-place state machine
OPEN = .80
CLOSE = -.05
HOVER = .075
OBSERVE_Q = np.array([-1.2, 0., 0., 0., 0., OPEN])


class PickPlaceTask:
    def __init__(self, model, data, robot, command, destination, box, event, source=None, vision=None):
        self.m, self.d, self.robot = model, data, robot
        self.command, self.box, self.event = command, box, event
        self.object_id = command["object_id"]
        self.destination = np.asarray(destination, dtype=float)
        self.vision = vision
        self.source = np.asarray(source).copy() if source is not None else data.body(self.object_id).xpos.copy()
        self.next_visual_check = 0.
        self.visual_good = False
        self.visual_evidence = None
        self.last_verify_xyz = None
        self.wrist_checked = set()
        self.cube_geom = model.geom(self.object_id + "_geom").id
        self.fixed_body = model.body("gripper").id
        self.moving_body = model.body("moving_jaw_so101_v1").id
        self.cube_v = model.jnt_dofadr[model.joint(self.object_id + "_free").id]
        support = command.get("target_id", "") + "_bottom" if box else "floor"
        self.support_geom = model.geom(support).id
        self.warning_baseline = data.warning.number.copy()
        self.max_lift = 0.
        self.lift_verified = self.done = self.failed = False
        self.reason = ""
        self.plans = []
        grasp_z = float(self.source[2] + .010)
        place_z = float(self.destination[2] + .026)
        specs = [
            ("APPROACH", [*self.source[:2], HOVER], OPEN),
            ("DESCEND", [*self.source[:2], grasp_z], OPEN),
            ("GRASP", [*self.source[:2], grasp_z], CLOSE),
            ("LIFT", [*self.source[:2], HOVER], CLOSE),
            ("TRANSFER", [*self.destination[:2], HOVER], CLOSE),
            ("LOWER", [*self.destination[:2], place_z], CLOSE),
            # Release with a small opening, withdraw above walls, then open fully.
            ("RELEASE", [*self.destination[:2], place_z], .25),
            ("RETREAT", [*self.destination[:2], HOVER], .25),
            ("OPEN", [*self.destination[:2], HOVER], OPEN),
            ("VERIFY", [*self.destination[:2], HOVER], OPEN),
        ]
        if vision:
            specs.insert(-1, ("OBSERVE", robot.tcp_at(OBSERVE_Q), OPEN))
            specs[-1] = ("VERIFY", robot.tcp_at(OBSERVE_Q), OPEN)
        previous_q = data.ctrl[robot.a].copy()
        rotation = data.body("gripper").xmat.reshape(3, 3)
        previous_xyz = data.site("tcp").xpos.copy() if np.linalg.norm(rotation[:, 2] - [0, 0, 1]) < .02 else None
        for name, xyz, aperture in specs:
            xyz = np.asarray(xyz)
            if vision and name in ("OBSERVE", "VERIFY"):
                points = np.array([previous_q, OBSERVE_Q])
            elif previous_xyz is None:
                points = np.array([previous_q, np.r_[robot.ik(xyz), aperture]])
            else:
                points = []
                seed = previous_q[:5]
                for s in np.linspace(0, 1, 21):
                    seed = robot.ik(previous_xyz + s * (xyz - previous_xyz), seed)
                    points.append(np.r_[seed, previous_q[5] + s * (aperture - previous_q[5])])
                points = np.asarray(points)
            if np.any(points < robot.limits[:, 0]) or np.any(points > robot.limits[:, 1]):
                raise ValueError(f"Joint limits exceeded in {name}")
            slope = float(np.max(np.abs(np.diff(points, axis=0)))) * (len(points) - 1)
            duration = max(2., 1.5 * slope / .30)
            self.plans.append((name, xyz, points, 0. if name == "VERIFY" else duration))
            previous_xyz, previous_q = xyz, points[-1]
        self.index = -1
        self.next_phase()

    def contacts(self):
        fixed = moving = support = False
        force = np.zeros(6)
        for i, contact in enumerate(self.d.contact):
            if self.cube_geom not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, force)
            if force[0] < .005:
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            fixed |= bool(self.m.geom_bodyid[other] == self.fixed_body)
            moving |= bool(self.m.geom_bodyid[other] == self.moving_body)
            support |= bool(other == self.support_geom)
        return fixed, moving, support

    def details(self):
        fixed, moving, support = self.contacts()
        if self.vision:
            return dict(object_id=self.object_id, source_xyz_m=np.round(self.source, 5).tolist(),
                        destination_xyz_m=np.round(self.destination, 5).tolist(),
                        fixed_contact=fixed, moving_contact=moving, support_contact=support,
                        tcp_lift_m=round(self.max_lift, 5), visual_evidence=self.visual_evidence)
        cube = self.d.body(self.object_id).xpos.copy()
        return dict(object_id=self.object_id, cube_xyz_m=np.round(cube, 5).tolist(),
                    destination_xyz_m=np.round(self.destination, 5).tolist(),
                    target_error_xy_m=round(float(np.linalg.norm(cube[:2] - self.destination[:2])), 5),
                    fixed_contact=fixed, moving_contact=moving, support_contact=support,
                    max_lift_m=round(self.max_lift, 5))

    def next_phase(self):
        self.index += 1
        if self.index == len(self.plans):
            self.done = True
            return
        self.phase, self.xyz, self.points, self.duration = self.plans[self.index]
        self.elapsed = self.stable = self.lost_contact = 0.
        self.event(self.command["id"], "RUNNING", phase=self.phase)

    def fail(self, code, reason):
        self.failed = True
        self.reason = code
        recoverable = self.vision is not None and code == "PREGRASP_VISION"
        self.event(self.command["id"], "RUNNING" if recoverable else "FAIL", phase=self.phase,
                   error=code, reason=reason, recoverable=recoverable, details=self.details())

    def tick(self, dt):
        if self.done or self.failed:
            return
        if not np.all(np.isfinite(self.d.qpos)) or not np.all(np.isfinite(self.d.qvel)) or np.any(self.d.warning.number > self.warning_baseline):
            self.fail("NUMERICAL_ERROR", "Nonfinite state or MuJoCo warning")
            return
        fixed, moving, support = self.contacts()
        held = fixed and moving
        cube = None if self.vision else self.d.body(self.object_id).xpos
        lift = float(self.d.site("tcp").xpos[2] - (self.source[2] + .010)) if self.vision else float(cube[2] - self.source[2])
        if held or not self.vision:
            self.max_lift = max(self.max_lift, lift)
        if self.phase == "TRANSFER":
            self.lost_contact = 0. if held else self.lost_contact + dt
            if self.lost_contact > .25 or (not self.vision and cube[2] < self.source[2] + .025):
                self.fail("OBJECT_LOST", "Object lost during transfer")
                return
        self.elapsed += dt
        u = min(1., self.elapsed / self.duration) if self.duration else 1.
        s = (3 * u * u - 2 * u * u * u) * (len(self.points) - 1)
        j = min(int(s), len(self.points) - 2)
        blend = s - j
        self.d.ctrl[self.robot.a] = (1 - blend) * self.points[j] + blend * self.points[j + 1]
        target = self.points[-1]
        arm_error = float(np.max(np.abs(self.d.qpos[self.robot.q[:5]] - target[:5])))
        arm_speed = float(np.max(np.abs(self.d.qvel[self.robot.v[:5]])))
        tcp_error = float(np.linalg.norm(self.d.site("tcp").xpos - self.xyz))
        ready = self.elapsed >= self.duration and arm_error < .03 and arm_speed < .10 and tcp_error < .004
        if self.vision and self.phase in ("APPROACH", "DESCEND") and ready and self.phase not in self.wrist_checked:
            try:
                self.visual_evidence = self.vision.wrist_confirm(self.d, self.object_id, self.source)
                self.wrist_checked.add(self.phase)
            except ValueError as exc:
                self.fail("PREGRASP_VISION" if self.phase == "APPROACH" else "FINAL_GRASP_CHECK", str(exc))
                return
        if self.phase in ("APPROACH", "DESCEND", "RELEASE", "RETREAT", "OPEN", "VERIFY"):
            ready &= abs(self.d.qpos[self.robot.q[5]] - target[5]) < .03
        if self.phase == "DESCEND" and not self.vision:
            ready &= np.linalg.norm(cube[:2] - self.source[:2]) < .008
        if self.phase == "GRASP":
            ready &= held
        if self.phase in ("LIFT", "TRANSFER"):
            ready &= held and lift > .035
            if not self.vision:
                ready &= np.linalg.norm(cube[:2] - self.xyz[:2]) < .01
        if self.phase == "LOWER":
            ready &= held
            if not self.vision:
                ready &= abs(cube[2] - (self.destination[2] + HALF)) < .005
        if self.phase == "VERIFY" and self.vision:
            if self.d.time >= self.next_visual_check:
                self.next_visual_check = self.d.time + .2
                try:
                    objects, boxes = self.vision.scan(self.d)
                    point = objects[self.object_id]["xyz"]
                    good = np.linalg.norm(point[:2] - self.destination[:2]) < .012
                    if self.box:
                        box_xy = boxes[self.command["target_id"]]["xyz"][:2]
                        good &= np.all(np.abs(point[:2] - box_xy) + HALF < np.array(BOX_INNER) - .001)
                    good &= self.last_verify_xyz is not None and np.linalg.norm(point - self.last_verify_xyz) < .003
                    self.last_verify_xyz = point.copy()
                    self.visual_good = bool(good)
                    self.visual_evidence = {"camera": "overhead_cam", "stamp": self.vision.stamp,
                                            "cube_xyz_m": np.round(point, 5).tolist(), "inside_target": bool(good)}
                except ValueError as exc:
                    self.visual_good = False
                    self.last_verify_xyz = None
                    self.visual_evidence = {"error": str(exc)}
            ready &= self.visual_good and self.lift_verified and support and not fixed and not moving
        elif self.phase == "VERIFY":
            velocity = self.d.qvel[self.cube_v:self.cube_v + 6]
            ready &= self.lift_verified and support and not fixed and not moving
            ready &= np.linalg.norm(cube[:2] - self.destination[:2]) < .012
            ready &= np.linalg.norm(velocity[:3]) < .01 and np.linalg.norm(velocity[3:]) < .10
            ready &= self.d.body(self.object_id).xmat.reshape(3, 3)[2, 2] > .95
            ready &= contained(self.d, self.object_id, self.box) if self.box else abs(cube[2] - HALF) < .003
        self.stable = self.stable + dt if ready else 0.
        if self.stable >= (1. if self.phase == "VERIFY" else .35):
            if self.phase == "LIFT":
                self.lift_verified = True
            self.event(self.command["id"], "RUNNING", phase=self.phase, phase_result="SUCCESS", details=self.details())
            self.next_phase()
        elif self.elapsed > self.duration + 5.:
            code = "GRASP_FAILED" if self.phase == "GRASP" else "FEEDBACK_TIMEOUT"
            self.fail(code, f"arm_error={arm_error:.4f} rad, tcp_error={tcp_error:.4f} m")


# Task dispatch
class TaskManager:
    def __init__(self, model, data, objects, boxes, hub):
        self.m, self.d, self.objects, self.boxes, self.hub = model, data, objects, boxes, hub
        self.robot = Robot(model)
        self.active = None

    def destination(self, command):
        if "target_xyz" in command:
            point = np.array(command["target_xyz"], dtype=float)
            # This simple free-placement demo has no obstacle planner.
            for box in self.boxes.values():
                if np.all(np.abs(point[:2] - box["xy"]) < [.08, .06]):
                    raise ValueError("TARGET_BLOCKED: use target_id to place inside a box")
            for name in self.objects:
                if name != command["object_id"] and np.linalg.norm(self.d.body(name).xpos[:2] - point[:2]) < .045:
                    raise ValueError("TARGET_OCCUPIED")
            return point, None
        box = self.boxes[command["target_id"]]
        for offset in (-.022, .022):
            point = np.array([box["xy"][0] + offset, box["xy"][1], BOX_FLOOR_Z])
            occupied = any(name != command["object_id"] and np.linalg.norm(self.d.body(name).xpos[:2] - point[:2]) < .037 for name in self.objects)
            if not occupied:
                return point, box
        raise ValueError("BOX_FULL: both placement slots are occupied")

    def start(self, command):
        object_id = command["object_id"]
        body = self.d.body(object_id)
        if body.xmat.reshape(3, 3)[2, 2] < .98:
            raise ValueError("OBJECT_TILTED: this baseline expects upright cubes")
        if "target_id" in command and contained(self.d, object_id, self.boxes[command["target_id"]]):
            self.hub.event(command["id"], "SUCCESS", phase="ALREADY_PLACED", reason="Object already in requested box")
            return
        if not HALF - .004 < body.xpos[2] < HALF + BOX_FLOOR_Z + .005:
            raise ValueError("OBJECT_NOT_RESTING: settle the object before sending a task")
        point, box = self.destination(command)
        self.hub.event(command["id"], "PLANNING", destination_xyz_m=point.tolist())
        # Precompute and validate the whole route BEFORE sending any new ctrl.
        self.active = PickPlaceTask(self.m, self.d, self.robot, command, point, box, self.hub.event)

    def tick(self, dt):
        if self.active:
            self.active.tick(dt)
            if self.active.failed:
                self.hub.latch_fault(self.active.reason)
                self.active = None
            elif self.active.done:
                self.hub.event(self.active.command["id"], "SUCCESS", phase="DONE", details=self.active.details())
                self.active = None
        elif self.hub.fault is None:
            command = self.hub.pop()
            if command:
                try:
                    self.start(command)
                except ValueError as exc:
                    self.hub.event(command["id"], "FAIL", phase="PLANNING", error="PLAN_REJECTED", reason=str(exc))
        with self.hub.lock:
            self.hub.telemetry = {
                "sim_time": round(self.d.time, 3),
                "active_id": self.active.command["id"] if self.active else None,
                "phase": self.active.phase if self.active else "IDLE",
                "objects": {name: np.round(self.d.body(name).xpos, 5).tolist() for name in self.objects},
                "bins": {name: [obj for obj in self.objects if contained(self.d, obj, box)] for name, box in self.boxes.items()},
            }


def queue_demo(hub, prefix="demo"):
    for number, obj, box in ((1, "red_cube", "left_box"), (2, "blue_cube", "right_box"), (3, "yellow_cube", "left_box")):
        hub.submit({"id": f"{prefix}-{number}", "action": "pick_place", "object_id": obj, "target_id": box})


class ObserveMotion:
    """Empty gripper moves to a known observation pose using joint feedback."""
    def __init__(self, data, robot, command):
        self.d, self.robot, self.command = data, robot, command
        self.start = data.ctrl[robot.a].copy()
        self.duration = max(2., 1.5 * np.max(np.abs(OBSERVE_Q - self.start)) / .30)
        self.elapsed = self.stable = 0.
        self.done = self.failed = False
        self.reason = ""
        self.phase = "OBSERVE"

    def tick(self, dt):
        if not np.all(np.isfinite(self.d.qpos)) or not np.all(np.isfinite(self.d.qvel)):
            self.failed, self.reason = True, "NUMERICAL_ERROR"
            return
        self.elapsed += dt
        u = min(1., self.elapsed / self.duration)
        self.d.ctrl[self.robot.a] = self.start + (3*u*u - 2*u*u*u) * (OBSERVE_Q - self.start)
        ready = (u == 1 and np.max(np.abs(self.d.qpos[self.robot.q] - OBSERVE_Q)) < .03
                 and np.max(np.abs(self.d.qvel[self.robot.v])) < .1)
        self.stable = self.stable + dt if ready else 0.
        self.done = self.stable >= .35
        if not self.done and self.elapsed > self.duration + 5:
            self.failed, self.reason = True, "OBSERVE_MOTION_TIMEOUT"


class VisualTaskManager:
    """Image-based task dispatch with bounded pre-grasp reobservation.

    Object and box coordinates come exclusively from VisionSystem. Joint/TCP
    feedback and jaw/support contacts remain simulator sensor adapters.
    """
    def __init__(self, model, data, objects, boxes, hub, vision):
        self.m, self.d, self.hub, self.vision = model, data, hub, vision
        self.robot = Robot(model)
        self.active = None
        self.attempt = 0
        self.started = self.scan_deadline = self.next_scan = 0.
        self.last_scan_error = ""
        self.scan_previous = None
        self.scan_stable = 0

    def observe(self, command):
        self.active = ObserveMotion(self.d, self.robot, command)
        self.scan_deadline = 0.
        self.scan_previous = None
        self.scan_stable = 0
        self.hub.event(command["id"], "RUNNING", phase="OBSERVE", attempt=self.attempt)

    def supported(self, object_id, support_name):
        pair = {self.m.geom(object_id + "_geom").id, self.m.geom(support_name).id}
        force = np.zeros(6)
        for i, contact in enumerate(self.d.contact):
            if {contact.geom1, contact.geom2} == pair and contact.efc_address >= 0:
                mujoco.mj_contactForce(self.m, self.d, i, force)
                if force[0] > .005:
                    return True
        return False

    def destination(self, command, objects, boxes):
        if "target_xyz" in command:
            point = np.array(command["target_xyz"], dtype=float)
            if any(np.all(np.abs(point[:2] - b["xyz"][:2]) < [.08, .06]) for b in boxes.values()):
                raise ValueError("TARGET_BLOCKED: use target_id for boxes")
            if any(n != command["object_id"] and np.linalg.norm(o["xyz"][:2] - point[:2]) < .045
                   for n, o in objects.items()):
                raise ValueError("TARGET_OCCUPIED")
            return point, None
        box = boxes[command["target_id"]]
        for offset in (-.022, .022):
            point = box["xyz"].copy()
            point[0] += offset
            if not any(n != command["object_id"] and np.linalg.norm(o["xyz"][:2] - point[:2]) < .037
                       for n, o in objects.items()):
                return point, {"xy": box["xyz"][:2].copy()}
        raise ValueError("BOX_FULL")

    def plan_from_images(self, command):
        objects, boxes = self.vision.scan(self.d)
        points = np.array([x["xyz"] for x in (*objects.values(), *boxes.values())])
        stable = self.scan_previous is not None and np.max(np.abs(points - self.scan_previous)) < .003
        self.scan_stable = self.scan_stable + 1 if stable else 1
        self.scan_previous = points
        if self.scan_stable < 3:
            raise ValueError("OBSERVATION_SETTLING: require three consistent RGB observations")
        source = objects[command["object_id"]]["xyz"]
        containing = [n for n, b in boxes.items() if np.all(np.abs(source[:2] - b["xyz"][:2]) + HALF
                                                           < np.array(BOX_INNER) - .001)]
        support_name = containing[0] + "_bottom" if containing else "floor"
        if not self.supported(command["object_id"], support_name):
            raise ValueError("OBJECT_NOT_SUPPORTED: known-plane localization requires a resting object")
        if "target_id" in command:
            inside = np.all(np.abs(source[:2] - boxes[command["target_id"]]["xyz"][:2]) + HALF
                            < np.array(BOX_INNER) - .001)
            if inside:
                self.hub.event(command["id"], "SUCCESS", phase="ALREADY_PLACED",
                               perception=self.vision.telemetry())
                self.active = None
                return
        point, box = self.destination(command, objects, boxes)
        self.hub.event(command["id"], "PLANNING", source="rgb_plane_geometry",
                       source_xyz_m=source.tolist(), destination_xyz_m=point.tolist(),
                       image_stamp=self.vision.stamp, attempt=self.attempt)
        self.active = PickPlaceTask(self.m, self.d, self.robot, command, point, box,
                                    self.hub.event, source=source, vision=self.vision)

    def tick(self, dt):
        if self.active:
            task = self.active
            if self.d.time - self.started > 100.:
                self.hub.event(task.command["id"], "FAIL", error="TASK_TIME_BUDGET", phase=task.phase)
                self.hub.latch_fault("TASK_TIME_BUDGET")
                self.active = None
            elif isinstance(task, ObserveMotion):
                task.tick(dt)
                if task.failed:
                    self.hub.event(task.command["id"], "FAIL", error=task.reason, phase=task.phase)
                    self.hub.latch_fault(task.reason)
                    self.active = None
                elif task.done:
                    if not self.scan_deadline:
                        self.scan_deadline = self.d.time + 2.
                        self.next_scan = self.d.time
                        task.phase = "LOCALIZE"
                    if self.d.time >= self.next_scan:
                        self.next_scan = self.d.time + .2
                        try:
                            self.plan_from_images(task.command)
                        except ValueError as exc:
                            self.last_scan_error = str(exc)
                            if self.d.time >= self.scan_deadline:
                                self.hub.event(task.command["id"], "FAIL", phase="LOCALIZE",
                                               error="PERCEPTION_OR_PLAN_REJECTED", reason=str(exc),
                                               recovery="Inspect scene or provide a new command")
                                self.active = None
            else:
                task.tick(dt)
                if task.failed:
                    if task.reason == "PREGRASP_VISION" and self.attempt < 2:
                        self.attempt += 1
                        self.observe(task.command)
                    else:
                        if task.reason == "PREGRASP_VISION":
                            self.hub.event(task.command["id"], "FAIL", error="RELOCALIZE_EXHAUSTED",
                                           phase=task.phase, attempts=self.attempt + 1)
                        self.hub.latch_fault(task.reason)
                        self.active = None
                elif task.done:
                    self.hub.event(task.command["id"], "SUCCESS", phase="DONE", details=task.details())
                    self.active = None
        elif self.hub.fault is None:
            command = self.hub.pop()
            if command:
                self.attempt = 0
                self.started = self.d.time
                self.observe(command)
        with self.hub.lock:
            perception = self.vision.telemetry()
            self.hub.telemetry = {"sim_time": round(self.d.time, 3),
                                  "active_id": self.active.command["id"] if self.active else None,
                                  "phase": self.active.phase if self.active else "IDLE",
                                  "objects": perception["objects"], "perception": perception,
                                  "bins": {box: [name for name, obj in self.vision.objects.items()
                                                 if np.all(np.abs(obj["xyz"][:2] - b["xyz"][:2]) + HALF
                                                           < np.array(BOX_INNER) - .001)]
                                           for box, b in self.vision.boxes.items()}}
