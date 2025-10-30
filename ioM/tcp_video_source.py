# ioM/tcp_video_source.py (auto-reconnect)
import socket, struct, numpy as np, cv2, time

class TCPVideoSource:
    def __init__(self, host, port, reconnect_delay=1.0):
        self.host, self.port = host, port
        self.reconnect_delay = reconnect_delay
        self.sock = None
        self._connect()

    def _connect(self):
        while True:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                print("[INFO] Connected to Pi camera stream")
                return
            except OSError as e:
                print(f"[WARN] connect failed: {e}; retry in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)

    def _recvall(self, n):
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def read(self):
        # try read header
        hdr = self._recvall(4)
        if hdr is None:
            print("[TCP] No header received. Reconnecting ...")
            try: self.sock.close()
            except: pass
            self._connect()
            return None  # ให้ loop ฝั่ง app.py ข้ามเฟรมนี้ไป

        length = struct.unpack(">I", hdr)[0]
        payload = self._recvall(length)
        if payload is None:
            print("[TCP] No payload received. Reconnecting ...")
            try: self.sock.close()
            except: pass
            self._connect()
            return None

        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        return frame

    def release(self):
        try: self.sock.close()
        except: pass