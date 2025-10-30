import cv2

class VideoSource:
    def __init__(self, src, w=None, h=None):
        self.cap = cv2.VideoCapture(src)
        if w: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        try: self.cap.release()
        except: pass
