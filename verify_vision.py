"""RGB integration checks. Linux without a desktop: MUJOCO_GL=egl python verify_vision.py.

Simulator object poses are used only as a test oracle and to inject perturbations.
The runtime vision module receives RGB plus camera calibration/pose, never those poses.
"""
import json

import mujoco
import numpy as np

from control import OBSERVE_Q, PickPlaceTask, Robot, VisualTaskManager, contained, queue_demo
from run_demo import CameraViews, CommandHub
from scene import build, ensure_model
from vision import VisionError, VisionSystem


def report(name, **details):
    print(json.dumps({"test": name, "passed": True, **details}), flush=True)


def create(path, shifted=False):
    m, objects, boxes = build(path, shifted)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, objects, boxes


def command(key="test"):
    return {"id": key, "action": "pick_place", "object_id": "red_cube", "target_id": "left_box"}


def step(m, d, manager):
    manager.tick(m.opt.timestep)
    mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)


def finish(m, d, manager, hub, hook=None):
    deadline = d.time + 230
    while (hub.queue or manager.active) and d.time < deadline:
        if hook:
            hook(manager)
        step(m, d, manager)
    assert not hub.queue and manager.active is None, "Simulation time budget exceeded"


def main():
    path = ensure_model()
    m, d, objects, boxes = create(path)
    robot = Robot(m)
    d.qpos[robot.q] = OBSERVE_Q
    rng = np.random.default_rng(23)
    with CameraViews(m, full_resolution=True, display=False) as cameras:
        vision = VisionSystem(m, cameras)
        worst = 0.
        for trial in range(5):
            for name, spec in objects.items():
                adr = m.jnt_qposadr[m.joint(name + "_free").id]
                d.qpos[adr:adr+2] = np.array(spec["xy"]) + rng.uniform(-.008, .008, 2)
            mujoco.mj_forward(m, d)
            detected, bins = vision.scan(d)
            for name, detection in {**detected, **bins}.items():
                error = np.linalg.norm(detection["xyz"][:2] - d.body(name).xpos[:2])
                worst = max(worst, float(error))
                assert error < .003, (name, error)
        report("rgb_localization_random_positions", trials=5, max_xy_error_m=round(worst, 6))

        # An old frame or a blank image must not fall back to simulator coordinates.
        try:
            vision.require_fresh(d.time + 1)
            raise AssertionError("Stale RGB was accepted")
        except VisionError:
            pass
        f = vision.frames["overhead_cam"]
        original = f.rgb
        f.rgb = np.zeros_like(f.rgb)
        try:
            vision.locate(f, "red_cube")
            raise AssertionError("Blank RGB was accepted")
        except VisionError:
            pass
        f.rgb = original.copy()
        detection = vision.locate(f, "red_cube")
        x, y, w, h = detection["bbox"]
        f.rgb[y+25:y+25+h, x:x+w] = original[y:y+h, x:x+w]
        try:
            vision.locate(f, "red_cube")
            raise AssertionError("Two red candidates were accepted")
        except VisionError as exc:
            assert "AMBIGUOUS" in str(exc), exc
        report("stale_blank_and_ambiguous_rgb_rejected")

    for shifted in (False, True):
        m, d, objects, boxes = create(path, shifted)
        events = []
        with CameraViews(m, True, False) as cameras:
            vision = VisionSystem(m, cameras)
            hub = CommandHub(objects, boxes, events.append)
            manager = VisualTaskManager(m, d, objects, boxes, hub, vision)
            # Deliberately poison the supplied layout dictionaries. They must not
            # be used as perception inputs by the visual controller.
            for spec in (*objects.values(), *boxes.values()):
                spec["xy"] = [9., 9.]
            queue_demo(hub)
            finish(m, d, manager, hub)
            assert all(t["status"] == "SUCCESS" for t in hub.records.values()), hub.snapshot()
            for name, target in (("red_cube", "left_box"), ("blue_cube", "right_box"), ("yellow_cube", "left_box")):
                assert contained(d, name, {"xy": d.body(target).xpos[:2]})
            assert all(e.get("source") == "rgb_plane_geometry" for e in events if e["status"] == "PLANNING")
            report("visual_sort_shifted" if shifted else "visual_sort_default", sim_time=round(d.time, 3))

    # Move the target after planning, while the arm is still approaching.
    m, d, objects, boxes = create(path)
    events = []
    moved = False
    with CameraViews(m, True, False) as cameras:
        vision = VisionSystem(m, cameras)
        hub = CommandHub(objects, boxes, events.append)
        manager = VisualTaskManager(m, d, objects, boxes, hub, vision)
        hub.submit(command("moved"))

        def move_target(manager):
            nonlocal moved
            if not moved and isinstance(manager.active, PickPlaceTask) and manager.active.phase == "APPROACH":
                adr = m.jnt_qposadr[m.joint("red_cube_free").id]
                d.qpos[adr:adr+2] += [.015, .025]
                mujoco.mj_forward(m, d)
                moved = True

        finish(m, d, manager, hub, move_target)
        assert moved and any(e.get("error") == "PREGRASP_VISION" for e in events), events
        assert hub.records["moved"]["status"] == "SUCCESS", hub.snapshot()
        assert "error" not in hub.records["moved"], hub.records["moved"]
        assert contained(d, "red_cube", {"xy": d.body("left_box").xpos[:2]})
        report("target_moves_reobserve_and_retry", sim_time=round(d.time, 3))

    # Remove the source color from both RGB streams; no grasp phase may start.
    m, d, objects, boxes = create(path)
    with CameraViews(m, True, False) as cameras:
        capture = cameras.capture
        cameras.capture = lambda data: {name: np.zeros_like(rgb) for name, rgb in capture(data).items()}
        events = []
        hub = CommandHub(objects, boxes, events.append)
        manager = VisualTaskManager(m, d, objects, boxes, hub, VisionSystem(m, cameras))
        hub.submit(command("occluded"))
        finish(m, d, manager, hub)
        assert hub.records["occluded"]["status"] == "FAIL"
        assert not any(e.get("phase") == "GRASP" for e in events)
        report("occluded_scene_fails_without_grasp")

    # A target moved during descent must be rejected before closing the gripper.
    m, d, objects, boxes = create(path)
    moved = False
    events = []
    with CameraViews(m, True, False) as cameras:
        hub = CommandHub(objects, boxes, events.append)
        manager = VisualTaskManager(m, d, objects, boxes, hub, VisionSystem(m, cameras))
        hub.submit(command("late-move"))

        def move_during_descent(manager):
            nonlocal moved
            if not moved and isinstance(manager.active, PickPlaceTask) and manager.active.phase == "DESCEND":
                adr = m.jnt_qposadr[m.joint("red_cube_free").id]
                d.qpos[adr+1] += .045
                mujoco.mj_forward(m, d)
                moved = True

        finish(m, d, manager, hub, move_during_descent)
        assert moved and hub.records["late-move"]["status"] == "FAIL", events
        assert any(e.get("error") == "FINAL_GRASP_CHECK" for e in events), events
        assert not any(e.get("phase") == "GRASP" for e in events)
        report("late_target_motion_stops_before_closing")

    m, d, objects, boxes = create(path)
    disabled = False
    with CameraViews(m, True, False) as cameras:
        hub = CommandHub(objects, boxes, lambda event: None)
        manager = VisualTaskManager(m, d, objects, boxes, hub, VisionSystem(m, cameras))
        hub.submit(command("fault"))
        hub.submit(command("after-fault"))

        def disable_gripper(manager):
            nonlocal disabled
            if not disabled and isinstance(manager.active, PickPlaceTask) and manager.active.phase == "GRASP":
                m.actuator_forcerange[m.actuator("gripper").id] = 0
                disabled = True

        finish(m, d, manager, hub, disable_gripper)
        assert disabled and hub.records["fault"]["status"] == "FAIL"
        assert hub.records["after-fault"]["status"] == "CANCELLED"
        report("visual_control_gripper_fault_cancels_queue")
    report("ALL_VISION_CHECKS_PASSED")


if __name__ == "__main__":
    main()
