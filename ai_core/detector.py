# ai_core/detector.py
# Ready-to-use YOLO detector with Lazy-Rotate (fast) and box mapping back to original frame.
from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class DetectorConfig:
    model_path: str = "yolo11s.pt"
    device: Optional[str] = None        # "cuda:0", "mps", or None
    imgsz: int = 640                    # 640 is a good fast default
    conf: float = 0.20
    iou: float = 0.50
    classes: Optional[List[int]] = None # e.g. [39] for bottle
    agnostic_nms: bool = True
    rotate_cooldown_frames: int = 4     # how many frames to wait before next rotate try
    enable_lazy_rotate: bool = True     # set False if you don't want rotation fallback


def _xyxy_to_int(box: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _map_box_from_rot90cw_to_original(x1, y1, x2, y2, H, W):
    """
    Map a bbox (xyxy) from a frame rotated 90° CW back to the original frame.
    For 90° CW: (x', y') in rotated corresponds to (x = y', y = W - 1 - x') in original.
    We map all 4 corners then min/max.
    """
    pts_rot = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float32)

    # x = y', y = W-1 - x'
    xs = pts_rot[:, 1]
    ys = (W - 1) - pts_rot[:, 0]
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    x_min, y_min, x_max, y_max = _clip_xyxy(x_min, y_min, x_max, y_max, W, H)
    return x_min, y_min, x_max, y_max


class Detector:
    """
    Drop-in detector:
      det = Detector(cfg.model)
      dets = det.infer(frame)
    Returns a list of dicts: { 'cls', 'conf', 'xyxy': (x1,y1,x2,y2) }
    """
    def __init__(self, cfg_or_model):
        # Accept either a path string or a namespace-like config with .fields
        if isinstance(cfg_or_model, str):
            cfg = DetectorConfig(model_path=cfg_or_model, classes=[39])
        elif isinstance(cfg_or_model, dict):
            cfg = DetectorConfig(
                model_path=cfg_or_model.get("model_path", "yolo11s.pt"),
                device=cfg_or_model.get("device", None),
                imgsz=int(cfg_or_model.get("imgsz", 640)),
                conf=float(cfg_or_model.get("conf", 0.20)),
                iou=float(cfg_or_model.get("iou", 0.50)),
                classes=cfg_or_model.get("classes", [39]),
                agnostic_nms=bool(cfg_or_model.get("agnostic_nms", True)),
                rotate_cooldown_frames=int(cfg_or_model.get("rotate_cooldown_frames", 4)),
                enable_lazy_rotate=bool(cfg_or_model.get("enable_lazy_rotate", True)),
            )
        else:
            # assume object with attributes (e.g., OmegaConf)
            cfg = DetectorConfig(
                model_path=getattr(cfg_or_model, "model_path", "yolo11s.pt"),
                device=getattr(cfg_or_model, "device", None),
                imgsz=int(getattr(cfg_or_model, "imgsz", 640)),
                conf=float(getattr(cfg_or_model, "conf", 0.20)),
                iou=float(getattr(cfg_or_model, "iou", 0.50)),
                classes=getattr(cfg_or_model, "classes", [39]),
                agnostic_nms=bool(getattr(cfg_or_model, "agnostic_nms", True)),
                rotate_cooldown_frames=int(getattr(cfg_or_model, "rotate_cooldown_frames", 4)),
                enable_lazy_rotate=bool(getattr(cfg_or_model, "enable_lazy_rotate", True)),
            )

        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.rotate_cooldown = 0

    def _predict(self, frame) -> Any:
        return self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=self.cfg.classes,
            agnostic_nms=self.cfg.agnostic_nms,
            device=self.cfg.device,
            augment=False,
            verbose=False
        )[0]

    def _to_dets(self, result, W, H) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return dets

        b = result.boxes
        xyxy = b.xyxy.cpu().numpy()            # (N,4)
        cls = b.cls.cpu().numpy().astype(int)  # (N,)
        conf = b.conf.cpu().numpy()            # (N,)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = _xyxy_to_int(xyxy[i])
            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
            dets.append({
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "xyxy": (x1, y1, x2, y2)
            })
        return dets

    def _to_dets_from_rot90cw(self, result, W, H) -> List[Dict[str, Any]]:
        """Map boxes from 90° CW rotated frame back to original (W,H)."""
        dets: List[Dict[str, Any]] = []
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return dets

        b = result.boxes
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()

        # In the rotated frame, width/height are swapped: (H_rot, W_rot) = (W, H)
        for i in range(len(xyxy)):
            x1r, y1r, x2r, y2r = _xyxy_to_int(xyxy[i])
            x1, y1, x2, y2 = _map_box_from_rot90cw_to_original(x1r, y1r, x2r, y2r, H, W)
            dets.append({
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "xyxy": (x1, y1, x2, y2)
            })
        return dets

    def infer(self, frame) -> List[Dict[str, Any]]:
        """
        Main detection API used by app.py
        Returns list of dicts with keys: 'cls', 'conf', 'xyxy'
        """
        H, W = frame.shape[:2]

        # Pass 1: original orientation
        res0 = self._predict(frame)
        dets0 = self._to_dets(res0, W, H)
        if dets0:
            # reset cooldown since we found something
            self.rotate_cooldown = max(0, self.rotate_cooldown - 1)
            return dets0

        # If nothing and lazy-rotate enabled, try one rotated pass with cooldown
        if self.cfg.enable_lazy_rotate:
            if self.rotate_cooldown <= 0:
                f90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                res90 = self._predict(f90)
                dets90 = self._to_dets_from_rot90cw(res90, W, H)
                # set cooldown whether found or not (to avoid too frequent rotations)
                self.rotate_cooldown = self.cfg.rotate_cooldown_frames
                if dets90:
                    return dets90
            else:
                self.rotate_cooldown -= 1

        # Nothing found
        return []
