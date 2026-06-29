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
# 1️⃣ Camera Intrinsics (카메라 내부 파라미터)
# =====================================================
# fx, fy : 초점거리 (pixel)
# cx, cy : principal point (이미지 중심)
FX, FY = 800.0, 800.0
CX, CY = 480.0, 270.0

# 카메라 투영 행렬 (3D → 2D)
P2 = np.array([
    [FX, 0,  CX, 0],
    [0,  FY, CY, 0],
    [0,   0,  1,  0]
], dtype=np.float32)

# =====================================================
# 2️⃣ 차량 크기 (실제 물리 단위, meters)
# =====================================================
CAR_WIDTH  = 2.0   # 차량 폭 (좌우)
CAR_LENGTH = 5.0   # 차량 길이 (전후)
CAR_HEIGHT = 1.7   # 차량 높이

# =====================================================
# 3️⃣ 단안 거리 추정
# Z = f * H / h_pixel
# =====================================================
def estimate_distance_from_bbox(y1, y2):
    """
    bbox의 세로 픽셀 크기를 이용한 단안 거리 추정
    """
    h_pixel = y2 - y1
    if h_pixel <= 0:
        return None

    # 단안 카메라 기본 수식
    Z = FY * CAR_HEIGHT / h_pixel
    return Z


def estimate_yaw_from_bbox(x1, x2, y1, y2):
    """
    bbox 가로/세로 비율을 이용한 간단한 yaw 추정
    (정확하지 않지만 BEV 시각화용으로 충분)
    """
    aspect = (x2 - x1) / max((y2 - y1), 1)
    yaw = (aspect - 1.0) * 0.6

    # yaw 범위 제한 (rad)
    return np.clip(yaw, -0.8, 0.8)

# =====================================================
# 4️⃣ YOLO 모델 로딩
# =====================================================
def load_yolo_model(path="yolov8n.pt"):
    """
    YOLOv8 모델 로딩
    """
    model = YOLO(path)
    print("✅ YOLOv8 loaded!")
    return model

# =====================================================
# 5️⃣ Detection + 3D 계산 + BEV 생성
# =====================================================
def detect_objects(model, frame):
    """
    입력 이미지에서
    - 차량 검출
    - 거리(Z), 위치(X), 방향(yaw) 추정
    - 3D 박스 계산
    - BEV 생성
    """
    results = model(frame)
    bev_boxes = []   # BEV에 전달할 차량 정보 리스트

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # 차량 클래스만 사용
            if model.names[cls] not in ["car", "truck", "bus"]:
                continue

            # ===============================
            # YOLO 2D Bounding Box
            # ===============================
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # ===============================
            # 1️⃣ 거리(Z) 추정
            # ===============================
            Z = estimate_distance_from_bbox(y1, y2)
            if Z is None:
                continue

            # ===============================
            # 2️⃣ Pixel → Camera 좌표 변환
            # ===============================
            # bbox 중심을 차량 중심으로 가정
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            X = (u - CX) / FX * Z
            Y = (v - CY) / FY * Z

            # ===============================
            # 3️⃣ Yaw 추정
            # ===============================
            yaw = estimate_yaw_from_bbox(x1, x2, y1, y2)

            # ===============================
            # 4️⃣ 이미지에 YOLO 2D 박스 표시
            # ===============================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Z={Z:.1f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # ===============================
            # 5️⃣ 3D Bounding Box 계산
            # ===============================
            corners_3d = compute_3d_box_corners(
                X, Y, Z,
                CAR_WIDTH,
                CAR_HEIGHT,
                CAR_LENGTH,
                yaw
            )

            # 3D → 이미지 평면 투영
            corners_2d = project_corners_to_image(corners_3d, P2)

            # 이미지에 3D 박스 시각화 (하늘색)
            #frame = draw_3d_box(frame, corners_2d)

            # ===============================
            # 6️⃣ BEV용 데이터 저장
            # ===============================
            bev_boxes.append({
                "X": X,               # 좌우 위치 (m)
                "Z": Z,               # 전방 거리 (m)
                "W": CAR_WIDTH,       # 차량 폭
                "L": CAR_LENGTH,      # 차량 길이
                "yaw": yaw            # 차량 방향
            })

    # ===============================
    # 7️⃣ BEV 생성 및 출력
    # ===============================
    bev = bev_from_detections(bev_boxes)
    cv2.imshow("BEV Map", bev)

    return frame

# =====================================================
# 6️⃣ Main (이미지 1장 입력)
# =====================================================
def main():
    # YOLO 로딩
    model = load_yolo_model("yolov8n.pt")

    # 입력 이미지
    img_path = "/home/kjs/Downloads/p1.png"
    frame = cv2.imread(img_path)

    if frame is None:
        print("❌ Image load error")
        return

    # Detection + BEV
    out = detect_objects(model, frame)

    # 결과 출력
    cv2.imshow("3D Object Detection (Pixel Distance)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

