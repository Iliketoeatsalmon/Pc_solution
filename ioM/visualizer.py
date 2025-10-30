import cv2

def draw(frame, target, text):
    if target:
        x1,y1,x2,y2 = target["xyxy"]
        cx = frame.shape[1]//2
        cy = frame.shape[0]//2
        cy_obj = (y1+y2)//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
        cv2.circle(frame, (target["obj_x"], cy_obj), 4, (0,0,255), -1)
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

def safe_show(title, frame, enable=True):
    if not enable: return
    try:
        cv2.imshow(title, frame); cv2.waitKey(1)
    except cv2.error:
        pass
