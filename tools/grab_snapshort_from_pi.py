import socket, struct, numpy as np, cv2, os

PI_IP = "192.168.1.103"   # confirm this
PORT = 6000
OUT = "tools/calib_from_pi.jpg"
os.makedirs("tools", exist_ok=True)

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)

with socket.create_connection((PI_IP, PORT), timeout=5) as s:
    hdr = recv_exact(s, 4)
    if not hdr:
        raise RuntimeError("No header bytes (4) received")
    (length,) = struct.unpack(">I", hdr)
    payload = recv_exact(s, length)
    if not payload:
        raise RuntimeError(f"Incomplete JPEG payload ({length} expected)")

arr = np.frombuffer(payload, np.uint8)
frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if frame is None:
    raise RuntimeError("cv2.imdecode failed")
cv2.imwrite(OUT, frame)
print("✅ Saved snapshot:", OUT, "size:", frame.shape)
