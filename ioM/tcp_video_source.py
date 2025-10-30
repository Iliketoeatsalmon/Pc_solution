# ioM/tcp_video_source.py
import cv2, socket, struct, numpy as np

class TCPVideoSource:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port))
        self.buf = b""

    def read(self):
        hdr = self._recvall(4)
        if not hdr:
            print("[TCP] No header received.")
            return None
        length = struct.unpack(">I", hdr)[0]
        data = self._recvall(length)
        if not data:
            print("[TCP] No data received.")
            return None
        print(f"[TCP] Got frame: {len(data)} bytes")
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _recvall(self, size):
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def release(self):
        try:
            self.sock.close()
        except:
            pass