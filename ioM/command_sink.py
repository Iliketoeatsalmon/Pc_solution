import json

class CommandSink:
    def __init__(self, print_cmd=True):
        self.print_cmd = print_cmd
        # ภายหลัง: สร้าง TCP ไปหา Pi ได้ในคลาสนี้

    def send(self, obj: dict):
        if self.print_cmd:
            print("CMD:", json.dumps(obj, separators=(",",":")))
        # ภายหลัง: ส่งผ่าน socket.sendall(...)

    def close(self): pass
