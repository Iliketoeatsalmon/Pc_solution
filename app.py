# app.py (เวอร์ชันถูกต้อง พร้อมระบบ homography + search)
import time
from utils.cfg import load_cfg
# from ioM.video_source import VideoSource
from ioM.command_sink import CommandSink
from ioM.visualizer import draw, safe_show
from ioM.tcp_video_source import TCPVideoSource
from ai_core.detector import Detector
from ai_core.controller import make_command
from ai_core.postprocess import load_H, pick_best_target_fused


def make_search_cmd():
    """หมุนหาเป้าหมาย: สลับทิศทุก ๆ 4 วินาที พร้อมขยับหน้าเบา ๆ"""
    period = 4.0
    dir_sign = 1 if int(time.time() // period) % 2 == 0 else -1
    vx = 0.10
    wz = 0.40 * dir_sign
    return {"type": "SET_TWIST",
            "data": {"vx": vx, "wz": wz, "distance_cm": -1}}, \
        f"SEARCH vx={vx:.2f} wz={wz:.2f}"


def main():
    cfg = load_cfg("config.yaml")

    USE_PI_CAMERA = True
    if USE_PI_CAMERA:
        src = TCPVideoSource("192.168.1.103", 6000)  # <-- ใส่ IP ของ Pi
        print("[INFO] Using Pi camera stream")
    else:
        # src = VideoSource(cfg.camera.source, cfg.camera.width, cfg.camera.height)
        print("[INFO] Using local webcam")

    # 1) I/O
    # src = VideoSource(cfg.camera.source, cfg.camera.width, cfg.camera.height)
    sink = CommandSink(print_cmd=cfg.runtime.print_cmd)

    # 2) Model
    det = Detector(cfg.model)

    # 3) Homography
    try:
        H = load_H("H.npy")
        print("[INFO] Homography loaded: H.npy")
    except Exception as e:
        H = None
        print(f"[WARN] Cannot load H.npy ({e}); fallback to width-based distance")

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue

            # --- รัน YOLO ---
            dets = det.infer(frame)

            # --- เลือก target แบบ fused (homography + width) ---
            target = pick_best_target_fused(
                dets,
                cfg.classes["bottle"],
                frame.shape[1],
                frame.shape[0],
                H,
                cfg.geometry.real_object_width_cm,
                cfg.geometry.focal_length_px,
                cfg.geometry.h_fov_deg,
                y_ratio_ground=0.90,
            )

            # --- ตัดสินใจคำสั่ง ---
            if target is None:
                # ไม่มีเป้าหมาย → SEARCH mode
                cmd, overlay = make_search_cmd()
            else:
                # มีเป้าหมาย → ใช้ PID logic เดิม (vx,wz)
                cmd, overlay = make_command(target, cfg.control)

            # --- ส่งคำสั่ง ---
            sink.send(cmd)

            # --- แสดงผล ---
            draw(frame, target, overlay)
            safe_show("AI Desktop", frame, enable=cfg.runtime.gui)

    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    main()