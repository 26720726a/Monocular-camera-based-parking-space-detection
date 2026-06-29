import cv2
import numpy as np
import time
from ultralytics import YOLO

# ===============================
# BEV 및 3D 유틸 함수
# ===============================
from bev_map import bev_from_detections

# =====================================================
# 1️⃣ Camera Intrinsics
# =====================================================
FX, FY = 800.0, 800.0
CX, CY = 480.0, 270.0

# =====================================================
# 2️⃣ 차량 크기 (meters)
# =====================================================
CAR_WIDTH  = 2.0
CAR_LENGTH = 5.0
CAR_HEIGHT = 1.7

# =====================================================
# 🔧 기준 차량 실제 거리
# =====================================================
REF_DISTANCE_M = 18.0

# =====================================================
# 🔥 기준 스케일 (1회 고정)
# =====================================================
REF_SCALE = None
REF_INITIALIZED = False

# =====================================================
# 🔥 Kalman Filters (Z only)
# =====================================================
KALMAN_Z = {}   # key → Kalman1D instance


# =====================================================
# Kalman Filter (1D)
# =====================================================
class Kalman1D:
    def __init__(self, Q=0.05, R=0.5):
        self.x = None
        self.P = 1.0
        self.Q = Q
        self.R = R

    def update(self, z):
        if self.x is None:
            self.x = z
            return z

        # Predict
        P_pred = self.P + self.Q
        x_pred = self.x

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


# =====================================================
# 거리 추정
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
# 기준 차량 선택 (오른쪽 + 가장 가까움)
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
# YOLO 로딩
# =====================================================
def load_yolo_model(path="yolov8n.pt"):
    model = YOLO(path)
    print("✅ YOLOv8 loaded!")
    return model


# =====================================================
# Detection + Kalman + BEV
# =====================================================
def detect_objects(model, frame):
    global REF_SCALE, REF_INITIALIZED

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
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "u": u, "v": v,
                "Z_raw": Z_raw
            })

    # -------------------------------
    # 기준 차량 1회 초기화
    # -------------------------------
    if not REF_INITIALIZED and len(detections_tmp) > 0:
        ref = select_reference_vehicle(detections_tmp)
        if ref:
            REF_SCALE = REF_DISTANCE_M / ref["Z_raw"]
            REF_INITIALIZED = True
            print(f"[INIT] REF_SCALE = {REF_SCALE:.3f}")

    if not REF_INITIALIZED:
        return frame

    # -------------------------------
    # Kalman Filter 적용
    # -------------------------------
    for d in detections_tmp:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        u, v = d["u"], d["v"]

        Z_meas = d["Z_raw"] * REF_SCALE

        key = int(u // 30)   # 간이 ID (tracking 대용)

        if key not in KALMAN_Z:
            KALMAN_Z[key] = Kalman1D()

        Z = KALMAN_Z[key].update(Z_meas)

        X = (u - CX) / FX * Z
        yaw = estimate_yaw_from_bbox(x1, x2, y1, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"Z={Z:.1f}m",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

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
# Main
# =====================================================
def main():
    model = load_yolo_model("yolov8n.pt")

    cap = cv2.VideoCapture("/home/kjs/Downloads/vison.webm")
    if not cap.isOpened():
        print("❌ Video open failed")
        return

    sim_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        target_time = sim_start_time + video_time

        out = detect_objects(model, frame)
        cv2.imshow("3D Object Detection (Video)", out)

        sleep = target_time - time.time()
        if sleep > 0:
            time.sleep(sleep)

        if cv2.waitKey(1) in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

