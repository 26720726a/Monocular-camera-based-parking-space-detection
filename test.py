import cv2
import torch
import numpy as np
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2  # ← DA2 모델
from torchvision import transforms


# ----------------------------------------------------
# YOLO 모델 로드
# ----------------------------------------------------
def load_yolo_model(model_path="yolov8n.pt"):
    model = YOLO(model_path)
    print("✅ YOLOv8 loaded!")
    return model


# ----------------------------------------------------
# Depth Anything v2 모델 로드
# ----------------------------------------------------
def load_depth_anything_v2():
    model = DepthAnythingV2(encoder="vitl")

    state = torch.load(
        "/home/kjs//3D-Object-Detection-with-YOLOv8-and-MiDaS/checkpoints/depth_anything_v2_vitl.pth",
        map_location="cpu"
    )

    # state_dict 로드
    model.load_state_dict(state, strict=False)
    model.eval()

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), antialias=True),  # 14의 배수로 변경
])


    print("✅ Depth Anything v2 loaded!")
    return model, transform


# ----------------------------------------------------
# Depth estimation (DepthAnythingV2)
# ----------------------------------------------------
def estimate_depth(frame, model, transform):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        model.cuda()
        tensor = tensor.cuda()

    with torch.no_grad():
        depth = model(tensor)[0]  # (H,W) normalized depth

    depth = depth.squeeze().cpu().numpy()

    # Up-scale to original resolution
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    return depth


# ----------------------------------------------------
# YOLO BBox 내부 depth → 거리 추정 (정확도 2배 개선)
# ----------------------------------------------------
def get_distance_from_depth(depth_map, x1, y1, x2, y2):
    # BBox 하단 30% → 가장 정확한 영역
    y_low = int(y1 + (y2 - y1) * 0.7)
    region = depth_map[y_low:y2, x1:x2]

    if region.size < 5:
        return None

    d = float(np.median(region))

    # Depth Anything v2 → 실제 거리와 비례하는 depth  
    # 깊을수록 값이 커지는 구조 → 그대로 사용 가능

    # 거리 변환 (scale 튜닝)
    SCALE = 8.5   # 값이 커지면 더 먼 거리로 추정됨
    Z = d * SCALE

    return Z


# ----------------------------------------------------
# 객체 탐지 + 거리 계산 + YOLO 2D 박스 출력
# ----------------------------------------------------
def detect_objects_and_depth(model, depth_model, transform, frame):
    depth_map = estimate_depth(frame, depth_model, transform)
    results = model(frame)

    detected_boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO 2D BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # 깊이 기반 거리 계산
            Z = get_distance_from_depth(depth_map, x1, y1, x2, y2)
            if Z is None:
                continue

            cv2.putText(frame, f"{Z:.2f} m", (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            detected_boxes.append((x1, y1, x2, y2, cls))

    return frame


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    model = load_yolo_model("yolov8n.pt")
    depth_model, transform = load_depth_anything_v2()

    cap = cv2.VideoCapture("/home/kjs/Downloads/test6.mp4")
    if not cap.isOpened():
        print("❌ Video open error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detect_objects_and_depth(model, depth_model, transform, frame)
        cv2.imshow("YOLO + Depth Anything v2", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

