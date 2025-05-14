import cv2
import numpy as np
import torch
from ultralytics import YOLO
from math import atan2, cos, sin, sqrt, pi

# — Draw an arrowed axis from origin in the direction of vec scaled by `scale` — #
def draw_axis(img, origin, vec, colour, thickness=2, scale=1.0):
    angle = atan2(vec[1], vec[0])
    length = sqrt(vec[0]*vec[0] + vec[1]*vec[1]) * scale
    end = (
        int(origin[0] + length * cos(angle)),
        int(origin[1] + length * sin(angle))
    )
    cv2.line(img, (int(origin[0]), int(origin[1])), end, colour, thickness)
    # arrow tip
    tip_size = 9
    for sign in (+1, -1):
        pt = (
            int(end[0] + tip_size * cos(angle + sign * pi/4)),
            int(end[1] + tip_size * sin(angle + sign * pi/4))
        )
        cv2.line(img, pt, end, colour, thickness)

# — Compute PCA on contour points and draw principal axes — #
def get_orientation(contour, img):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    origin = tuple(mean.astype(int))
    cv2.circle(img, origin, 3, (255, 0, 255), 2)
    draw_axis(img, origin, eigvecs[:,0] * eigvals[0], (0,255,0), scale=0.3)
    draw_axis(img, origin, eigvecs[:,1] * eigvals[1], (255,0,0), scale=0.3)

# — Overlay a semi-transparent filled mask and return its largest contour — #
def overlay_mask(frame, mask, colour=(255,200,100), alpha=0.5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    overlay = frame.copy()
    cv2.fillPoly(overlay, contours, colour)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    return max(contours, key=cv2.contourArea)

def main():
    # Load YOLOv8-Seg model (install via `pip install ultralytics`)
    model = YOLO('yolov8s-seg.pt')
    model.fuse()                   # fuse conv+bn for speed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)               # full FP32

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run segmentation (FP32)
        results = model(frame, device=device, imgsz=640)[0]

        masks = results.masks.data.cpu().numpy()       # (N,H,W)
        boxes = results.boxes.xyxy.cpu().numpy()       # (N,4)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for m, box, cls, conf in zip(masks, boxes, classes, confs):
            if conf < 0.5:
                continue

            bin_mask = (m > 0.5).astype(np.uint8)
            largest = overlay_mask(frame, bin_mask)   # draws mask
            if largest is not None and cv2.contourArea(largest) > 100:
                get_orientation(largest, frame)

            # bounding box + label
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"{model.names[cls]}: {conf*100:.0f}%"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow('YOLOv8-Seg + PCA', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
