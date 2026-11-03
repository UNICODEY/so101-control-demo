"""SO-101 sorting demo: JSON commands, task feedback and MuJoCo simulation."""
import argparse
import copy
import json
import math
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import mujoco
from control import CAMERAS, TaskManager, VisualTaskManager, build, queue_demo

MODEL_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "robotstudio_so101"


class CameraViews:
    """Two RGB streams from one physics state; rendering never steps physics.

    capture(data) returns {camera_name: RGB uint8 array}. Call only on the
    simulation thread. Each call captures both views at the same data.time.
    """
    def __init__(self, model, full_resolution=False, display=True):
        self.model, self.display = model, display
        self.renderers = {}
        self.windows = []
        self.cv2 = None
        self.sizes = {name: (spec["width"], spec["height"])
                      for name, spec in CAMERAS.items()}
        if not full_resolution:
            self.sizes["overhead_cam"] = (960, 540)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:5] = 0  # Show the actual visual meshes, not collision proxies.

    def __enter__(self):
        try:
            if self.display:
                import cv2
                self.cv2 = cv2
            for name, (width, height) in self.sizes.items():
                self.renderers[name] = mujoco.Renderer(self.model, height=height, width=width)
                if self.display:
                    self.cv2.namedWindow(name, self.cv2.WINDOW_NORMAL)
                    self.windows.append(name)
                    self.cv2.resizeWindow(name, min(width, 960), min(height, 540))
            if self.display:
                self.cv2.moveWindow("wrist_cam", 20, 40)
                self.cv2.moveWindow("overhead_cam", 680, 40)
            return self
        except Exception:
            self.close()
            raise

    def capture(self, data):
        frames = {}
        for name, renderer in self.renderers.items():
            renderer.update_scene(data, camera=name, scene_option=self.option)
            frames[name] = renderer.render().copy()
        return frames

    def show(self, data, phase, vision=None):
        # Check before imshow so closing a camera does not recreate its window.
        try:
            visible = all(self.cv2.getWindowProperty(name, self.cv2.WND_PROP_VISIBLE) >= 1
                          for name in self.windows)
        except self.cv2.error:
            visible = False
        if not visible:
            return False
        for name, rgb in self.capture(data).items():
            frame = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
            if name == "overhead_cam" and vision and 0 <= data.time - vision.stamp < .3:
                for object_id, detection in {**vision.objects, **vision.boxes}.items():
                    x, y, width, height = detection["bbox"]
                    self.cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    self.cv2.putText(frame, object_id, (x, y - 5), self.cv2.FONT_HERSHEY_SIMPLEX,
                                     .5, (255, 255, 255), 1, self.cv2.LINE_AA)
            self.cv2.putText(frame, f"{name} | {data.time:.2f}s | {phase}", (12, 25),
                             self.cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1,
                             self.cv2.LINE_AA)
            self.cv2.imshow(name, frame)
        return self.cv2.waitKey(1) & 0xff != 27

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
        if self.cv2:
            for name in self.windows:
                try:
                    self.cv2.destroyWindow(name)
                except self.cv2.error:
                    pass
        self.windows.clear()

    def __exit__(self, *exc):
        self.close()


def ensure_model():
    """Fetch only the pinned SO-101 XML, meshes and upstream license."""
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MODEL_COMMIT}/robotstudio_so101/"

    def download(relative):
        path = MODEL_DIR / relative
        if path.is_file() and path.stat().st_size:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        try:
            with urllib.request.urlopen(base + relative, timeout=60) as response:
                content = response.read()
            if not content:
                raise OSError("Empty model download: " + relative)
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    download("so101.xml")
    root = ET.parse(MODEL_DIR / "so101.xml").getroot()
    files = ["scene.xml", "LICENSE"]
    for mesh in root.findall("asset/mesh"):
        name = mesh.attrib["file"]
        if Path(name).name != name or "\\" in name:
            raise ValueError("Unexpected mesh path in upstream model")
        files.append("assets/" + name)
    if any(not (MODEL_DIR / name).is_file() for name in files):
        print("Downloading pinned SO-101 model (first run)...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    return MODEL_DIR / "scene.xml"


# Command interface
def console_event(payload):
    print(json.dumps(payload, ensure_ascii=True), flush=True)


class CommandHub:
    def __init__(self, object_ids, box_ids, emit=console_event):
        self.lock = threading.RLock()
        self.object_ids, self.box_ids = set(object_ids), set(box_ids)
        self.queue = deque()
        self.records = {}
        self.commands = {}
        self.fault = None
        self.telemetry = {}
        self.emit = emit

    def submit(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Command must be a JSON object")
        if set(payload) - {"id", "action", "object_id", "target_id", "target_xyz"}:
            raise ValueError("Unknown command fields")
        command = copy.deepcopy(payload)
        task_id = command.get("id")
        if isinstance(task_id, bool) or not isinstance(task_id, (str, int)) or not 0 < len(str(task_id)) <= 64:
            raise ValueError("id must be a nonempty string/integer, at most 64 characters")
        command["id"] = str(task_id)
        if command.get("action") != "pick_place":
            raise ValueError("Only action=pick_place is supported")
        if command.get("object_id") not in self.object_ids:
            raise ValueError("Unknown object_id")
        if ("target_id" in command) == ("target_xyz" in command):
            raise ValueError("Specify exactly one of target_id or target_xyz")
        if "target_id" in command and command["target_id"] not in self.box_ids:
            raise ValueError("Unknown target_id")
        if "target_xyz" in command:
            point = command["target_xyz"]
            if not isinstance(point, list) or len(point) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                raise ValueError("target_xyz must contain three finite numbers")
            if point[2] != 0:
                raise ValueError("Free placement currently supports the ground plane z=0 only")
        with self.lock:
            key = command["id"]
            if key in self.records:
                if self.commands[key] != command:
                    raise ValueError("ID_CONFLICT: this id already identifies a different command")
                return {"duplicate": True, **copy.deepcopy(self.records[key])}
            if self.fault:
                raise ValueError("FAULT_LATCHED: inspect the failure and restart the simulation")
            if len(self.queue) >= 32:
                raise ValueError("Queue full (32 commands)")
            self.commands[key] = command
            self.records[key] = {"id": key, "status": "QUEUED", "command": command}
            self.queue.append(command)
            self.emit({"id": key, "status": "QUEUED", "object_id": command["object_id"]})
            return copy.deepcopy(self.records[key])

    def pop(self):
        with self.lock:
            return self.queue.popleft() if self.queue and self.fault is None else None

    def event(self, task_id, status, **fields):
        with self.lock:
            record = self.records[str(task_id)]
            if fields.get("phase") != record.get("phase"):
                record.pop("phase_result", None)
            if status in ("RUNNING", "PLANNING", "SUCCESS") and "error" not in fields:
                for key in ("error", "reason", "recoverable"):
                    record.pop(key, None)
            record.update(status=status, **fields)
            self.emit({"id": str(task_id), "status": status, **fields})

    def latch_fault(self, reason):
        with self.lock:
            self.fault = reason
            # Keep the current actuator targets. Do not start queued movements.
            while self.queue:
                command = self.queue.popleft()
                self.event(command["id"], "CANCELLED", reason="Earlier execution failed")

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(dict(fault=self.fault,
                                      queue=[c["id"] for c in self.queue],
                                      tasks=list(self.records.values()), **self.telemetry))


def serve(hub, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self.reply(200, hub.snapshot())
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            try:
                if self.headers.get("Origin"):
                    raise ValueError("Browser-origin requests are not supported; use the included CLI")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Expected JSON body, maximum 16 KiB")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/commands":
                    self.reply(202, hub.submit(payload))
                else:
                    self.reply(404, {"error": "Unknown endpoint"})
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"status": "REJECTED", "reason": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# Simulation entry point
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="Existing scene.xml; omitted: download the pinned model")
    parser.add_argument("--manual", action="store_true", help="Start idle, waiting for API commands")
    parser.add_argument("--shifted", action="store_true", help="Use nearby alternative object and box positions")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--state-control", action="store_true", help="Explicit legacy comparison: use simulator object coordinates instead of RGB")
    parser.add_argument("--no-cameras", action="store_true", help="Show only the original 3D viewer")
    parser.add_argument("--camera-fps", type=float, default=15., help="RGB preview update rate, 1-30 (default 15)")
    parser.add_argument("--camera-full-res", action="store_true", help="Render overhead RGB at 1920x1080 instead of 960x540")
    parser.add_argument("--port", type=int, default=8765, help="Local API port; 0 disables HTTP")
    parser.add_argument("--log", type=Path, help="JSONL log path; default: logs/<timestamp>.jsonl")
    args = parser.parse_args()
    if not math.isfinite(args.camera_fps) or not 1 <= args.camera_fps <= 30:
        parser.error("--camera-fps must be between 1 and 30")
    if args.model is None:
        args.model = ensure_model()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}; use --model with the existing scene.xml")
    if not 0 <= args.port < 65536 or (args.headless and args.manual and args.port == 0):
        parser.error("Use a valid port; manual headless mode needs the API")
    log_path = args.log or Path(__file__).resolve().parent / "logs" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6] + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model, objects, boxes = build(args.model, args.shifted)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    log_lock = threading.Lock()
    server = None
    with ExitStack() as resources, log_path.open("w", encoding="utf-8") as log_file:
        def emit(event):
            line = json.dumps(event, ensure_ascii=True)
            with log_lock:
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

        hub = CommandHub(objects, boxes, emit)
        cameras = None
        if not args.state_control or (not args.headless and not args.no_cameras):
            cameras = resources.enter_context(CameraViews(model, not args.state_control or args.camera_full_res,
                                                          display=not args.headless and not args.no_cameras))
        if args.state_control:
            manager = TaskManager(model, data, objects, boxes, hub)
        else:
            from vision import VisionSystem
            manager = VisualTaskManager(model, data, objects, boxes, hub, VisionSystem(model, cameras))
        try:
            if args.port:
                server = serve(hub, args.port)
            emit({"event": "READY", "api": f"http://127.0.0.1:{args.port}" if args.port else None,
                  "log": str(log_path), "layout": "shifted" if args.shifted else "default",
                  "mode": "manual" if args.manual else "auto_demo",
                  "control_source": "simulator_state" if args.state_control else "rgb_plane_geometry"})
            if not args.manual:
                queue_demo(hub)
            print("SIMULATION ONLY | close viewer/camera window: stop | ESC in RGB window: stop", flush=True)
            print("Left box = green (+Y); right box = purple (-Y). These names use the robot frame.", flush=True)
            completed_announced = False

            def tick():
                nonlocal completed_announced
                manager.tick(model.opt.timestep)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                state = hub.snapshot()
                finished = state["tasks"] and not state["queue"] and manager.active is None
                if finished and not completed_announced:
                    emit({"event": "QUEUE_COMPLETE", "success": sum(t["status"] == "SUCCESS" for t in state["tasks"]),
                          "failed": sum(t["status"] == "FAIL" for t in state["tasks"]), "bins": state["bins"],
                          "sim_time": round(data.time, 3)})
                    completed_announced = True
                elif not finished:
                    completed_announced = False
                return finished

            if args.headless:
                while True:
                    finished = tick()
                    if finished and not args.manual:
                        break
                    if args.manual:
                        time.sleep(model.opt.timestep)
            else:
                import mujoco.viewer as viewer_module

                with viewer_module.launch_passive(model, data) as viewer:
                    with viewer.lock():
                        viewer.cam.lookat[:] = [.17, 0., .10]
                        viewer.cam.distance = .78
                        viewer.cam.azimuth = 135.
                        viewer.cam.elevation = -35.
                    viewer.sync()
                    next_frame = time.perf_counter()
                    next_camera = next_frame
                    if cameras:
                        print("RGB cameras: wrist follows gripper; overhead stays fixed. Ideal lenses, no depth.", flush=True)
                    while viewer.is_running():
                        started = time.perf_counter()
                        tick()
                        if started >= next_frame:
                            viewer.sync()
                            next_frame = started + .02
                        if cameras and cameras.display and started >= next_camera:
                            if not cameras.show(data, manager.active.phase if manager.active else "IDLE",
                                                getattr(manager, "vision", None)):
                                break
                            next_camera = time.perf_counter() + 1. / args.camera_fps
                        time.sleep(max(0., model.opt.timestep - (time.perf_counter() - started)))
        finally:
            if manager.active:
                hub.event(manager.active.command["id"], "CANCELLED", reason="Simulation stopped")
            with hub.lock:
                while hub.queue:
                    hub.event(hub.queue.popleft()["id"], "CANCELLED", reason="Simulation stopped")
            if server:
                server.shutdown()
                server.server_close()
        return 1 if hub.fault or any(t["status"] == "FAIL" for t in hub.records.values()) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}), flush=True)
        sys.exit(1)
