import cv2, numpy as np

IMG_SRC = "tools/calib_from_pi.jpg"   # เดิมเคยเป็น 0
OUT_NPY = "clicked_points.npy"


pts = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        print(f"clicked: {x},{y}")

def main():
    if isinstance(IMG_SRC, int):
        cap = cv2.VideoCapture(IMG_SRC)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Cannot read camera frame")
    else:
        img = cv2.imread(IMG_SRC)
        if img is None:
            raise RuntimeError("Cannot read image file")

    disp = img.copy()
    cv2.namedWindow("click 4 ground corners (clockwise)")
    cv2.setMouseCallback("click 4 ground corners (clockwise)", on_mouse)

    while True:
        d = disp.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(d, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(d, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("click 4 ground corners (clockwise)", d)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('s') and len(pts) == 4:
            np.save(OUT_NPY, np.array(pts, dtype=np.float32))
            print("saved:", OUT_NPY)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
