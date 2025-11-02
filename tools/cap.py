#!/usr/bin/env python3
# tools/capture_dataset.py
# PC-side viewer/saver for dataset collection from a Raspberry Pi camera stream

import sys, os, time, csv, pathlib
import cv2
import numpy as np
import socket, struct

# ============================================
# 🔧 CONFIG (ตั้งค่าได้ตรงนี้)
PI_IP = "192.168.195.177"    # ✅ ใส่ IP ของ Raspberry Pi ที่รันกล้องไว้
VIDEO_PORT = 6000           # ✅ พอร์ตของสตรีม (ต้องตรงกับฝั่ง Pi)
OUTPUT_DIR = "dataset_session"
AUTO_SAVE_INTERVAL = 0.0    # 0 = ปิดอัตโนมัติ, 1.0 = เซฟทุก 1 วินาที
WINDOW_TITLE = "Dataset Capture"
# ============================================


# ============ Helper: TCP video receiver ============
def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)

def recv_frame_tcp(sock):
    """รับ JPEG frame ผ่าน TCP ([4-byte length][payload])"""
    try:
        hdr = recv_exact(sock, 4)
        if not hdr:
            return None
        (length,) = struct.unpack(">I", hdr)
        payload = recv_exact(sock, length)
        if not payload:
            return None
        arr = np.frombuffer(payload, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# ============ Directory & Logging ============
def ensure_dirs(base):
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "bottle").mkdir(parents=True, exist_ok=True)
    (base / "leaf").mkdir(parents=True, exist_ok=True)

def timestamp_name(prefix="img"):
    t = time.localtime()
    ms = int((time.time() - int(time.time())) * 1000)
    return f"{prefix}_{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}_{ms:03d}.jpg"


# ============ Main ============
def main():
    print(f"[INFO] Connecting to Pi camera at {PI_IP}:{VIDEO_PORT} ...")
    sock = socket.create_connection((PI_IP, VIDEO_PORT), timeout=5)
    sock.settimeout(2.0)
    print("[OK] Connected to Pi video stream")

    out_dir = pathlib.Path(OUTPUT_DIR)
    ensure_dirs(out_dir)

    csv_path = out_dir / "labels.csv"
    new_file = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if new_file:
        csv_w.writerow(["filename", "class", "timestamp"])

    auto_on = AUTO_SAVE_INTERVAL > 0.0
    flip = False
    last_save_t = time.time()

    print("\n🎥 Controls:")
    print("  SPACE  -> save to raw/")
    print("  1      -> save to bottle/")
    print("  2      -> save to leaf/")
    print("  a      -> toggle auto-save")
    print("  f      -> flip left-right")
    print("  q      -> quit\n")

    try:
        while True:
            frame = recv_frame_tcp(sock)
            if frame is None:
                print("[WARN] Lost frame, retrying ...")
                time.sleep(0.05)
                continue

            if flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            hud = f"[a]uto:{'ON' if auto_on else 'OFF'} {AUTO_SAVE_INTERVAL:.1f}s  [f]lip:{'ON' if flip else 'OFF'}  [1]=bottle  [2]=leaf  [SPACE]=raw  [q]=quit"
            cv2.putText(frame, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            saved = False
            label = None

            # Auto save
            if auto_on and (time.time() - last_save_t) >= max(0.2, AUTO_SAVE_INTERVAL):
                fname = timestamp_name("auto")
                dst = out_dir / "raw" / fname
                cv2.imwrite(str(dst), frame)
                csv_w.writerow([str(dst.relative_to(out_dir)), "raw", f"{time.time():.3f}"])
                csv_f.flush()
                last_save_t = time.time()
                saved = True
                label = "raw"

            if key == ord('q'):
                break
            elif key == ord(' '):
                fname = timestamp_name("raw")
                dst = out_dir / "raw" / fname
                cv2.imwrite(str(dst), frame)
                csv_w.writerow([str(dst.relative_to(out_dir)), "raw", f"{time.time():.3f}"])
                csv_f.flush()
                saved = True
                label = "raw"
            elif key == ord('1'):
                fname = timestamp_name("bottle")
                dst = out_dir / "bottle" / fname
                cv2.imwrite(str(dst), frame)
                csv_w.writerow([str(dst.relative_to(out_dir)), "bottle", f"{time.time():.3f}"])
                csv_f.flush()
                saved = True
                label = "bottle"
            elif key == ord('2'):
                fname = timestamp_name("leaf")
                dst = out_dir / "leaf" / fname
                cv2.imwrite(str(dst), frame)
                csv_w.writerow([str(dst.relative_to(out_dir)), "leaf", f"{time.time():.3f}"])
                csv_f.flush()
                saved = True
                label = "leaf"
            elif key == ord('a'):
                auto_on = not auto_on
                last_save_t = time.time()
            elif key == ord('f'):
                flip = not flip

            if saved:
                print(f"[SAVE] {label}: {dst}")

    finally:
        csv_f.close()
        try: sock.close()
        except: pass
        cv2.destroyAllWindows()
        print("[INFO] Exit dataset capture")

if __name__ == "__main__":
    main()