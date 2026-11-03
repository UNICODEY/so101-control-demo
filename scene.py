"""Build the MuJoCo test environment; no motion control or task execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

MODEL_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "robotstudio_so101"


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


