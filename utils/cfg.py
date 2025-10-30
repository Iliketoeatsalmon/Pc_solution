# utils/cfg.py
from dataclasses import dataclass
from types import SimpleNamespace
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# ---------- Dataclasses ----------
@dataclass
class CameraCfg:
    source: int = 0
    width: int = 640
    height: int = 480

@dataclass
class GeometryCfg:
    focal_length_px: float = 115.0
    real_object_width_cm: float = 6.5
    h_fov_deg: float = 60.0

@dataclass
class ControlCfg:
    desired_time_s: float = 5.0
    stop_distance_cm: float = 55.0
    max_vx: float = 0.6
    wz_gain: float = 1.0
    # เพิ่มฟิลด์ใหม่ที่คุณใช้
    yaw_bias_deg: float = 0.0
    yaw_deadband_deg: float = 2.0
    yaw_min_wz: float = 0.12
    yaw_max_wz: float = 0.8

@dataclass
class DetectorCfg:
    imgsz: int = 640
    conf: float = 0.20
    iou: float = 0.50
    rotate_cd: int = 4
    enable_lazy_rotate: bool = True

@dataclass
class RuntimeCfg:
    gui: bool = True
    print_cmd: bool = True
    device: Optional[str] = None  # "cuda:0", "mps", "cpu", หรือ None

@dataclass
class AppCfg:
    # model: string path เช่น "yolo11s.pt"
    model: str
    classes: Dict[str, int]
    camera: CameraCfg
    geometry: GeometryCfg
    control: ControlCfg
    detector: DetectorCfg
    runtime: RuntimeCfg


# ---------- Helpers ----------
def _ns(obj: Any, **defaults):
    """return SimpleNamespace with defaults if obj is None"""
    if obj is None:
        return SimpleNamespace(**defaults)
    if isinstance(obj, dict):
        base = {**defaults, **obj}
        return SimpleNamespace(**base)
    return obj  # already namespace-like


# ---------- Main loader ----------
def load_cfg(path: str) -> AppCfg:
    path = Path(path)
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))

    # --- model path: รองรับทั้งแบบ string และแบบ {weights: "..."}
    model_field = data.get("model", "yolo11s.pt")
    if isinstance(model_field, str):
        model_path = model_field
    elif isinstance(model_field, dict):
        model_path = model_field.get("weights", "yolo11s.pt")
    else:
        model_path = "yolo11s.pt"

    # --- classes ---
    classes = data.get("classes", {"bottle": 39})

    # --- camera ---
    cam = _ns(data.get("camera"), source=0, width=640, height=480)
    camera = CameraCfg(source=int(cam.source), width=int(cam.width), height=int(cam.height))

    # --- geometry ---
    geo = _ns(
        data.get("geometry"),
        focal_length_px=115.0,
        real_object_width_cm=6.5,
        h_fov_deg=60.0,
    )
    geometry = GeometryCfg(
        focal_length_px=float(geo.focal_length_px),
        real_object_width_cm=float(geo.real_object_width_cm),
        h_fov_deg=float(geo.h_fov_deg),
    )

    # --- control (รองรับคีย์ใหม่ ๆ ที่คุณเพิ่ม) ---
    ctrl = _ns(
        data.get("control"),
        desired_time_s=5.0,
        stop_distance_cm=55.0,
        max_vx=0.6,
        wz_gain=1.0,
        yaw_bias_deg=0.0,
        yaw_deadband_deg=2.0,
        yaw_min_wz=0.12,
        yaw_max_wz=0.8,
    )
    control = ControlCfg(
        desired_time_s=float(ctrl.desired_time_s),
        stop_distance_cm=float(ctrl.stop_distance_cm),
        max_vx=float(ctrl.max_vx),
        wz_gain=float(ctrl.wz_gain),
        yaw_bias_deg=float(ctrl.yaw_bias_deg),
        yaw_deadband_deg=float(ctrl.yaw_deadband_deg),
        yaw_min_wz=float(ctrl.yaw_min_wz),
        yaw_max_wz=float(ctrl.yaw_max_wz),
    )

    # --- detector ---
    det = _ns(
        data.get("detector"),
        imgsz=640,
        conf=0.20,
        iou=0.50,
        rotate_cd=4,
        enable_lazy_rotate=True,
    )
    detector = DetectorCfg(
        imgsz=int(det.imgsz),
        conf=float(det.conf),
        iou=float(det.iou),
        rotate_cd=int(det.rotate_cd),
        enable_lazy_rotate=bool(det.enable_lazy_rotate),
    )

    # --- runtime ---
    rt = _ns(
        data.get("runtime"),
        gui=True,
        print_cmd=True,
        device=None,
    )
    runtime = RuntimeCfg(gui=bool(rt.gui), print_cmd=bool(rt.print_cmd), device=rt.device)

    return AppCfg(
        model=model_path,
        classes=classes,
        camera=camera,
        geometry=geometry,
        control=control,
        detector=detector,
        runtime=runtime,
    )
