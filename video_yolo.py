import cv2
import torch
import numpy as np
from ultralytics import YOLO
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from bev_map import bev_from_detections
from utils_3d import compute_3d_box_corners, project_corners_to_image, draw_3d_box


# ----------------------------------------------------
# Yaw estimation
# ----------------------------------------------------
def estimate_yaw_from_depth(depth_map, x1, y1, x2, y2):
    obj_region = depth_map[int(y1):int(y2), int(x1):int(x2)]
    if obj_region.size == 0:
        return 0.0

    h, w = obj_region.shape
    left_region = obj_region[:, :w // 4]
    right_region = obj_region[:, 3 * w // 4:]

    left_depth = np.median(left_region)
    right_depth = np.median(right_region)

    diff = right_depth - left_depth
    yaw = np.arctan2(diff, np.mean([left_depth, right_depth]))
    yaw = float(np.clip(yaw, -0.8, 0.8))
    return yaw


# ----------------------------------------------------
# Camera intrinsics
# ----------------------------------------------------
P2 = np.array([
    [800,   0, 480, 0],
    [  0, 800, 270, 0],
    [  0,   0,   1, 0]
], dtype=np.float32)


# ----------------------------------------------------
# YOLO model
# ----------------------------------------------------
def load_yolo_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    print("✅ YOLOv8 loaded!")
    return model


# ----------------------------------------------------
# MiDaS model
# ----------------------------------------------------
def load_midas_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()

    transform = Compose([
        Resize(384, 384,
               resize_target=False,
               keep_aspect_ratio=True,
               ensure_multiple_of=32,
               resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return midas, transform


# ----------------------------------------------------
# Depth estimation
# ----------------------------------------------------
def estimate_depth(frame, midas, transform):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sample = transform({"image": img_rgb})
    input_tensor = torch.from_numpy(sample["image"]).unsqueeze(0)

    if torch.cuda.is_available():
        midas.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        depth = midas(input_tensor)

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
    depth = cv2.normalize(depth, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    return depth


# ----------------------------------------------------
# Depth → Distance
# ----------------------------------------------------
def get_distance_from_depth(depth_map, x1, y1, x2, y2):
    y_low = int(y1 + (y2 - y1) * 0.4)
    region = depth_map[y_low:y2, x1:x2]

    if region.size < 5:
        return None

    d_raw = float(np.median(region))
    d_min = np.percentile(depth_map, 10)
    d_max = np.percentile(depth_map, 90)

    d = np.clip(d_raw, d_min, d_max)
    Z = (d_max - d) / (d_max - d_min) * 25.0
    return Z


# ----------------------------------------------------
# YOLO + MiDaS + BEV
# ----------------------------------------------------
def detect_objects_and_depth(model, midas, transform, frame):
    depth_map = estimate_depth(frame, midas, transform)
    results = model(frame)
    detected_boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            Z_raw = get_distance_from_depth(depth_map, x1, y1, x2, y2)
            if Z_raw is None:
                continue

            Z = 0.85 * Z_raw + 1.20
            cv2.putText(frame, f"{Z:.2f} m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)

            detected_boxes.append((x1, y1, x2, y2, cls))

    bev = bev_from_detections(depth_map, detected_boxes)
    cv2.imshow("BEV Map", bev)

    return frame


# ----------------------------------------------------
# Main (VIDEO INPUT)
# ----------------------------------------------------
def main():
    model = load_yolo_model("yolov8n.pt")
    midas, transform = load_midas_model()

    cap = cv2.VideoCapture("/home/kjs/Downloads/test6.mp4")
    if not cap.isOpened():
        print("❌ Video open error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detect_objects_and_depth(model, midas, transform, frame)
        cv2.imshow("3D Object Detection + 3D BOX", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

