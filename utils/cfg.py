from dataclasses import dataclass
import yaml, pathlib

@dataclass
class GeometryCfg:
    focal_length_px: float
    real_object_width_cm: float
    h_fov_deg: float

@dataclass
class ControlCfg:
    desired_time_s: float
    stop_distance_cm: float
    max_vx: float
    wz_gain: float

@dataclass
class CameraCfg:
    source: str | int
    width: int
    height: int

@dataclass
class RuntimeCfg:
    gui: bool
    print_cmd: bool

@dataclass
class AppCfg:
    model: str
    classes: dict
    camera: CameraCfg
    geometry: GeometryCfg
    control: ControlCfg
    runtime: RuntimeCfg

def load_cfg(path="config.yaml") -> AppCfg:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return AppCfg(
        model=data["model"],
        classes=data["classes"],
        camera=CameraCfg(**data["camera"]),
        geometry=GeometryCfg(**data["geometry"]),
        control=ControlCfg(**data["control"]),
        runtime=RuntimeCfg(**data["runtime"]),
    )
