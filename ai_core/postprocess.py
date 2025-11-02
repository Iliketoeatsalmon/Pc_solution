# postprocess.py
import math# ai_core/postprocess.py

import math, numpy as np

def estimate_distance_cm_width(box_w_px, real_w_cm, focal_px):
    return (real_w_cm * focal_px) / max(1, box_w_px)

def bottom_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, max(y1, y2))

def pixel_to_ground(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    q /= (q[2] + 1e-9)
    return float(q[0]), float(q[1])  # X, Y (meters)

def plane_distance_m(X, Y):
    return math.sqrt(X*X + Y*Y)

def pick_best_target_fused(
    dets,
    allowed_classes,             # <-- NEW: iterable ของคลาสที่อนุญาต (เช่น {bottle_id, leaf_id})
    frame_w,
    frame_h,
    H_or_None,
    real_w_cm,
    focal_px,
    h_fov_deg,
    y_ratio_ground=0.90,
    dhard_max_m=10.0,
    dsoft_max_w_m=2.0
):
    """
    เลือกเป้าหมายที่ 'ใกล้สุด' จากหลายคลาส (allowed_classes)
    พร้อมฟิวส์ระยะ (Homography ถ้าแตะพื้น/อยู่ล่างภาพ, ไม่งั้น width-based)
    คืนค่า dict ที่มี: cls, xyxy, obj_x, distance_cm, angle_deg, method
    """
    if isinstance(allowed_classes, (int, np.integer)):
        allowed = {int(allowed_classes)}
    else:
        allowed = set(int(c) for c in allowed_classes)

    best = None
    center_x = frame_w // 2
    idx = 0

    for d in dets:
        cls = int(d["cls"])
        if cls not in allowed:
            continue
        idx += 1
        x1, y1, x2, y2 = d["xyxy"]
        wpx = max(1, x2 - x1)
        obj_x, yb = bottom_center(x1, y1, x2, y2)
        dx_px = obj_x - center_x
        angle_deg = (dx_px / frame_w) * h_fov_deg

        # ระยะแบบ width-based
        dW_cm = estimate_distance_cm_width(wpx, real_w_cm, focal_px)
        d_final_cm = dW_cm
        method = "width"

        # ถ้ามี H และกล่องแตะ/ใกล้พื้น -> ใช้ H
        if H_or_None is not None and yb >= int(y_ratio_ground * frame_h):
            X, Y = pixel_to_ground(H_or_None, obj_x, yb)
            dH_m = plane_distance_m(X, Y)
            dH_cm = dH_m * 100.0
            d_final_cm = dH_cm
            method = "H"
            # sanity check: ถ้า H ให้ค่าไกลเว่อร์ แต่ width ดูมีเหตุผล -> ใช้ width
            if dH_m > dhard_max_m and (dW_cm / 100.0) < dsoft_max_w_m:
                d_final_cm = dW_cm
                method = "width_sanity"

        cand = {
            "idx": idx,
            "cls": cls,                         # <-- คืนคลาส
            "xyxy": (x1, y1, x2, y2),          # <-- คืนพิกัดกล่อง
            "obj_x": obj_x,
            "distance_cm": float(d_final_cm),
            "angle_deg": float(angle_deg),
            "method": method,
        }
        if best is None or cand["distance_cm"] < best["distance_cm"]:
            best = cand
    return best
