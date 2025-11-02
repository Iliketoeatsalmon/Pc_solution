import numpy as np, cv2

CLICKED_NPY = "clicked_points.npy"
H_OUT = "H.npy"

# ระยะจริงบนพื้น (เมตร) ปรับตามไซต์งาน
L = 3.0  # ความยาวแกน X
W = 2.0  # ความกว้างแกน Y

def main():
    img_pts = np.load(CLICKED_NPY).astype(np.float32)  # shape (4,2) ตามเข็มนาฬิกา
    # พิกัดพื้น (เมตร) ตามลำดับจุดที่คลิก
    world_pts = np.array([[0,0],
                          [L,0],
                          [L,W],
                          [0,W]], dtype=np.float32)

    # สร้างโฮโมกราฟีจาก image->ground
    H, mask = cv2.findHomography(img_pts, world_pts, method=0)
    np.save(H_OUT, H)
    print("H saved to", H_OUT)
    print(H)
 
if __name__ == "__main__":
    main()
