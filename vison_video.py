import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# BEV 및 3D 유틸 함수
# ===============================
from bev_map import bev_from_detections
from utils_3d import (
    compute_3d_box_corners,
    project_corners_to_image,
    draw_3d_box
)

# =====================================================
# 1️⃣ Camera Intrinsics
# =====================================================
FX, FY = 800.0, 800.0
CX, CY = 480.0, 270.0

P2 = np.array([
    [FX, 0,  CX, 0],
    [0,  FY, CY, 0],
    [0,   0,  1,  0]
], dtype=np.float32)

# =====================================================
# 2️⃣ 차량 크기 (meters)
# =====================================================
CAR_WIDTH  = 2.0
CAR_LENGTH = 5.0
CAR_HEIGHT = 1.7

# =====================================================
# 🔧 기준 차량 실제 거리
# =====================================================
REF_DISTANCE_M = 18.0   # 오른쪽 가장 가까운 차량 = 18m

# =====================================================
# 3️⃣ 단안 거리 추정
# =====================================================
def estimate_distance_from_bbox(y1, y2):
    h_pixel = y2 - y1
    if h_pixel <= 0:
        return None
    return FY * CAR_HEIGHT / h_pixel


def estimate_yaw_from_bbox(x1, x2, y1, y2):
    aspect = (x2 - x1) / max((y2 - y1), 1)
    yaw = (aspect - 1.0) * 0.6
    return np.clip(yaw, -0.8, 0.8)

# =====================================================
# 🔧 기준 차량 선택 (오른쪽 + 가장 가까움)
# =====================================================
def select_reference_vehicle(detections):
    if len(detections) == 0:
        return None

    detections = sorted(
        detections,
        key=lambda d: (-d["u"], -d["v"])
    )
    return detections[0]

# =====================================================
# 4️⃣ YOLO 로딩
# =====================================================
def load_yolo_model(path="yolov8n.pt"):
    model = YOLO(path)
    print("✅ YOLOv8 loaded!")
    return model

# =====================================================
# 5️⃣ Detection + BEV
# =====================================================
def detect_objects(model, frame):
    results = model(frame, verbose=False)

    detections_tmp = []
    bev_boxes = []

    # -------------------------------
    # YOLO Detection
    # -------------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in ["car", "truck", "bus"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            Z_raw = estimate_distance_from_bbox(y1, y2)
            if Z_raw is None:
                continue

            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            detections_tmp.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "u": u, "v": v,
                "Z_raw": Z_raw
            })

    # -------------------------------
    # 기준 차량 보정
    # -------------------------------
    ref = select_reference_vehicle(detections_tmp)
    scale = REF_DISTANCE_M / ref["Z_raw"] if ref else 1.0

    # -------------------------------
    # 최종 거리 + BEV
    # -------------------------------
    for d in detections_tmp:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        u, v = d["u"], d["v"]

        Z = d["Z_raw"] * scale
        X = (u - CX) / FX * Z
        Y = (v - CY) / FY * Z
        yaw = estimate_yaw_from_bbox(x1, x2, y1, y2)

        color = (0, 255, 0)
        if d is ref:
            color = (0, 255, 0)  # 기준 차량

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Z={Z:.1f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        bev_boxes.append({
            "X": X,
            "Z": Z,
            "W": CAR_WIDTH,
            "L": CAR_LENGTH,
            "yaw": yaw
        })

    bev = bev_from_detections(bev_boxes)
    cv2.imshow("BEV Map", bev)

    return frame

# =====================================================
# 6️⃣ Main (VIDEO) - 영상 시간 = 시뮬 시간 (정답)
# =====================================================
def main():
    import time

    model = load_yolo_model("yolov8n.pt")

    cap = cv2.VideoCapture("/home/kjs/Downloads/vison.webm")
    print("CAP OPEN:", cap.isOpened())
    if not cap.isOpened():
        print("❌ Video open failed")
        return

    sim_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🎯 현재 프레임의 "영상 시간" (ms → sec)
        video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 🎯 이 프레임이 도달해야 할 실제 시뮬 시간
        target_real_time = sim_start_time + video_time_sec

        # YOLO + BEV
        out = detect_objects(model, frame)
        cv2.imshow("3D Object Detection (Video)", out)

        # ⏳ 핵심: 영상 시간과 실제 시간 동기화
        now = time.time()
        sleep_time = target_real_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

