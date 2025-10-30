from ultralytics import YOLO

class Detector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def infer(self, frame):
        # return list of dict: {cls, x1,y1,x2,y2, conf}
        out = []
        results = self.model(frame, verbose=False)
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                out.append({"cls": cls, "xyxy": (x1,y1,x2,y2), "conf": conf})
        return out
