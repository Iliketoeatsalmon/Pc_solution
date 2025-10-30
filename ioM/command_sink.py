# ioM/command_sink.py — TCP sender to Pi (port 6001), auto-reconnect, JSON lines
import socket, json, time

class CommandSink:
    def __init__(self, host="192.168.1.103", port=6001, print_cmd=True, reconnect_delay=1.0):
        self.host = host
        self.port = port
        self.print_cmd = print_cmd
        self.reconnect_delay = reconnect_delay
        self.sock = None
        self._connect()

    def _connect(self):
        while True:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if self.print_cmd:
                    print(f"[INFO] Connected to Pi CMD at {self.host}:{self.port}")
                return
            except OSError as e:
                if self.print_cmd:
                    print(f"[WARN] CMD connect failed: {e}; retry in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)

    def send(self, obj):
        try:
            msg = json.dumps(obj, ensure_ascii=False) + "\n"
            if self.print_cmd:
                print("CMD:", msg.strip())
            self.sock.sendall(msg.encode("utf-8"))
        except OSError:
            # reconnect and retry once
            try:
                self.sock.close()
            except:
                pass
            self._connect()
            try:
                msg = json.dumps(obj, ensure_ascii=False) + "\n"
                self.sock.sendall(msg.encode("utf-8"))
            except OSError:
                pass

    def close(self):
        try:
            self.sock.close()
        except:
            pass