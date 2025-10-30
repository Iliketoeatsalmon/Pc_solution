# ai_core/controller.py
import math, numpy as np

def make_command(target, control_cfg):
    """คำนวณ vx, wz และระยะ เพื่อส่งคำสั่งให้ Pi"""
    if target is None:
        # ไม่มีเป้าหมาย — ส่ง STOP พร้อมระยะ -1
        return {"type":"STOP", "data":{"distance_cm": -1}}, "STOP (no target)"

    d_cm = target["distance_cm"]
    ang_deg = target["angle_deg"]

    # ถ้าเข้าใกล้เกิน limit → หยุด
    if d_cm < control_cfg.stop_distance_cm:
        return {"type":"STOP", "data":{"distance_cm": d_cm}}, f"STOP (close) d={d_cm:.1f}cm"

    # --- คำนวณความเร็วเชิงเส้น vx (m/s) ---
    vx = (d_cm / 100.0) / max(1e-6, control_cfg.desired_time_s)
    vx = float(min(vx, control_cfg.max_vx))

    # --- คำนวณความเร็วเชิงมุม wz (rad/s) ---
    err_rad = math.radians(ang_deg)
    wz = float(np.clip(err_rad * control_cfg.wz_gain, -1.0, 1.0))

    # --- ส่งออก JSON พร้อมระยะ ---
    return {
        "type": "SET_TWIST",
        "data": {
            "vx": vx,
            "wz": wz,
            "distance_cm": d_cm
        }
    }, f"GO vx={vx:.2f} wz={wz:.2f} d={d_cm:.1f}cm"