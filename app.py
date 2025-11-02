#!/usr/bin/env python3
# app_pc.py — PC side: receive video (TCP:6000) -> YOLO -> send binary cmd to Pi (TCP:6001)
# Detect BOTH classes: bottle=0, leaf=1 → send state=1 for bottle, 2 for leaf
# Reads configuration from config.yaml

import os, sys, time, socket, struct
import numpy as np
import cv2
import yaml

# ---------- Import local modules ----------
from ai_core.filters import Kalman1D
from ai_core.postprocess import pick_best_target_fused


VIDEO_RECV_TIMEOUT_S = 10.0   # ยืดเวลาเผื่อเฟรมเว้นช่วง
RECV_DEADLINE_S      = 10.0   # เดดไลน์สะสมต่อการอ่านบล็อคหนึ่ง (header/payload)

# ---------- Load config ----------
def load_cfg(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- Networking ----------
def set_sock_opts(s):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

def connect_with_retry(host, port, name):
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5)
            set_sock_opts(s)
            if name == "video":
                # ยืด timeout ของ video socket
                s.settimeout(VIDEO_RECV_TIMEOUT_S)
            print(f"[OK] Connected to {name} {host}:{port}")
            return s
        except OSError as e:
            print(f"[WARN] connect {name} failed: {e}; retry in 1s")
            time.sleep(1)

def recv_exact(sock, n):
    """อ่านให้ได้ n ไบต์แน่นอน; ทนต่อ socket.timeout โดยวนรอจนถึง deadline"""
    buf = bytearray()
    deadline = time.monotonic() + RECV_DEADLINE_S
    while len(buf) < n:
        # หมดเวลาแล้ว -> ยอมแพ้ให้ลูปหลักรีคอนเนคต์
        if time.monotonic() > deadline:
            return None
        try:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None  # peer ปิด
            buf += chunk
        except socket.timeout:
            # เงียบ ๆ แล้ววนต่อจนกว่าจะถึง deadline
            continue
        except OSError:
            return None
    return bytes(buf)

def recv_frame_tcp(sock):
    # 4 byte ความยาว
    hdr = recv_exact(sock, 4)
    if not hdr:
        return None
    (length,) = struct.unpack(">I", hdr)

    # payload JPEG
    payload = recv_exact(sock, length)
    if not payload:
        return None

    arr = np.frombuffer(payload, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def send_bytes(cmd_sock, speed_percent, angle_deg, state, pi_ip, cmd_port, verbose=True):
    """
    Send binary packet: [speed(1B)][angle(2B)][state(1B)] big-endian
    - speed_percent: 0–100
    - angle_deg: -180–180
    - state: 0 none, 1 bottle, 2 leaf
    """
    packet = struct.pack(">B h B",
                         int(max(0, min(100, speed_percent))),
                         int(max(-180, min(180, angle_deg))),
                         int(max(0, min(255, state))))
    try:
        cmd_sock.sendall(packet)
        if verbose:
            print(f"SEND bytes: speed={speed_percent:3d}%  angle={angle_deg:4d}°  state={state}")
        return cmd_sock
    except OSError:
        try:
            cmd_sock.close()
        except:
            pass
        cmd_sock = connect_with_retry(pi_ip, cmd_port, "cmd")
        cmd_sock.sendall(packet)
        return cmd_sock

# ---------- Utility ----------
def draw_box_and_centers(frame, cx, best):
    if not best or "xyxy" not in best:
        return
    x1, y1, x2, y2 = best["xyxy"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.circle(frame, (cx, frame.shape[0]//2), 4, (255,0,0), -1)
    cy_obj = (y1 + y2) // 2
    cv2.circle(frame, (best["obj_x"], cy_obj), 4, (0,0,255), -1)


def distance_to_speed_pct(dist_cm):
    if dist_cm < 50:   return 30
    elif dist_cm < 60: return 50
    elif dist_cm < 100: return int(50 + (dist_cm - 60) * (50/40))
    else: return 100

# ---------- Main ----------
def main():
    CFG = load_cfg("config.yaml")

    # --- Network ---
    PI_HOST = CFG["network"]["pi_ip"]
    VIDEO_PORT = CFG["network"]["video_port"]
    CMD_PORT = CFG["network"]["cmd_port"]

    # --- YOLO Model ---
    from ultralytics import YOLO
    model_path = CFG["model"]
    det_cfg = CFG["detector"]
    INFERENCE_SIZE = det_cfg["imgsz"]
    CONFIDENCE_THRESHOLD = det_cfg["conf"]

    device_pref = CFG["runtime"]["device"]
    from torch import cuda, backends
    device = device_pref or ("cuda:0" if cuda.is_available() else "mps" if getattr(backends, "mps", None) and backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = YOLO(model_path)
    try: model.to(device)
    except: pass

    # --- Geometry ---
    geom = CFG["geometry"]
    FOCAL_PX = geom["focal_length_px"]
    REAL_W_CM = geom["real_object_width_cm"]
    HFOV_DEG = geom["h_fov_deg"]

    # --- Homography ---
    H = None
    if CFG.get("homography", {}).get("use", False):
        H_path = CFG["homography"].get("file", "tools/H.npy")
        try:
            H = np.load(H_path)
            print(f"[INFO] Loaded H from {H_path}")
        except Exception as e:
            print(f"[WARN] Can't load H: {e}")

    # --- Kalman Filter ---
    filt_cfg = CFG.get("filter", {})
    kf = Kalman1D(
        x0=filt_cfg.get("kf_init_cm", 150.0),
        p0=filt_cfg.get("kf_init_var", 200.0),
        q=filt_cfg.get("kf_q", 2.0),
        r=filt_cfg.get("kf_r", 50.0)
    )

    # --- Socket connect ---
    video_sock = connect_with_retry(PI_HOST, VIDEO_PORT, "video")
    cmd_sock   = connect_with_retry(PI_HOST, CMD_PORT, "cmd")

    # --- Classes ---
    bottle_id = CFG["classes"]["bottle"]
    leaf_id   = CFG["classes"]["leaf"]

    SHOW_WINDOW = CFG["runtime"]["gui"]
    PROCESS_EVERY_N = CFG["runtime"].get("process_every_n", 2)
    WINDOW_NAME = "Desktop AI View"

    last_best = None
    frame_id = 0

    try:
        while True:
            frame = recv_frame_tcp(video_sock)
            if frame is None:
                print("[TCP] video lost, reconnecting ...")
                try: video_sock.close()
                except: pass
                video_sock = connect_with_retry(PI_HOST, VIDEO_PORT, "video")
                continue

            frame_id += 1
            Hh, Ww = frame.shape[:2]
            cx = Ww // 2
            best = last_best

            if frame_id % PROCESS_EVERY_N == 0:
                results = model(frame, imgsz=INFERENCE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
                dets = []
                for r in results:
                    for b in r.boxes:
                        dets.append({
                            "cls": int(b.cls[0]),
                            "xyxy": tuple(map(int, b.xyxy[0]))
                        })

                best = pick_best_target_fused(
                    dets,
                    allowed_classes={bottle_id, leaf_id},   # <-- ใช้ทั้ง bottle และ leaf
                    frame_w=Ww,
                    frame_h=Hh,
                    H_or_None=H,
                    real_w_cm=REAL_W_CM,
                    focal_px=FOCAL_PX,
                    h_fov_deg=HFOV_DEG
                )
                last_best = best
            if best is None:
                cmd_sock = send_bytes(cmd_sock, 0, 0, 0, PI_HOST, CMD_PORT)
                cv2.putText(frame, "NO TARGET", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                # --- Fuse + Kalman ---
                dist_cm = kf.update(best["distance_cm"])
                angle_deg = best["angle_deg"]
                speed_pct = distance_to_speed_pct(dist_cm)

                # --- Decide state ---
                cls = best.get("cls", bottle_id)
                state_val = 1 if cls == bottle_id else 2
                label = "bottle" if cls == bottle_id else "leaf"

                cmd_sock = send_bytes(cmd_sock, speed_pct, int(round(angle_deg)), state_val, PI_HOST, CMD_PORT)

                draw_box_and_centers(frame, cx, best)
                overlay = f"{label}: {dist_cm:5.1f}cm {speed_pct}% {angle_deg:5.1f}°"
                cv2.putText(frame, overlay, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    send_bytes(cmd_sock, 0, 0, 0, PI_HOST, CMD_PORT)
                    break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")

    finally:
        try: video_sock.close()
        except: pass
        try: cmd_sock.close()
        except: pass
        cv2.destroyAllWindows()
        print("[INFO] Clean exit")

if __name__ == "__main__":
    main()