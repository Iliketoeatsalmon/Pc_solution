import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ioM.tcp_video_source import TCPVideoSource
import cv2

PI_IP = "192.168.1.103"  # <-- ใส่ IP ของ Pi
PORT = 6000

src = TCPVideoSource(PI_IP, PORT)
frame = src.read()
cv2.imwrite("tools/calib_from_pi.jpg", frame)
print("✅ Saved snapshot: tools/calib_from_pi.jpg")
