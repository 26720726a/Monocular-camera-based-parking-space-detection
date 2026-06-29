import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision import transforms
import time

# =========================================================
# 1. Camera intrinsics (너의 값으로 조정 가능)
# =========================================================
fx = 800
fy = 800
cx = 640
cy = 360

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# =========================================================
# 2. Yaw smoothing (EMA)
# =========================================================
prev_yaw = {}
yaw_alpha = 0.4   # 부드럽게 조절 → 0.2~0.5 추천

def smooth_yaw(object_id, yaw_raw):
    global prev_yaw
    if object_id not in prev_yaw:
        prev_yaw[object_id] = yaw_raw
        return yaw_raw
    prev = prev_yaw[object_id]
    filtered = yaw_alpha * yaw_raw + (1 - yaw_alpha) * prev
    prev_yaw[object_id] = filtered
    return filtered

# =========================================================
# 3. visualDet3D 기반 projection_utils (축약 버전)
# =========================================================
def compute_box_3d(dim, location, yaw):
    """ dim = [h,w,l], location = [x,y,z] """
    h, w, l = dim
    x, y, z = location

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    R = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [          0, 1,          0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    corners = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners = corners + np.array(location).reshape(3,1)
    return corners.T

def project_to_image(pts_3d, K):
    pts_2d = K @ pts_3d.T
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2].T.astype(int)

def draw_3d_box(img, pts_2d, color=(255, 200, 0)):
    pts = pts_2d
    # 4 lower + 4 upper points
    for i, j in zip([0,1,2,3,0], [1,2,3,0,4]):
        cv2.line(img, pts[i], pts[j], color, 2)
    for i, j in zip([4,5,6,7,4], [5,6,7,4,0]):
        cv2.line(img, pts[i], pts[j], color, 2)
    for i in range(4):
        cv2.line(img, pts[i], pts[i+4], color, 2)

# =========================================================
# 4. Pixel → Camera 좌표 변환
# =========================================================
def pixel_to_camera(u, v, depth):
    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth
    return X, Y, Z

# =========================================================
# 5. Depth median from BBox
# =========================================================
def get_bbox_median_depth(depth_map, x1, y1, x2, y2):
    h = depth_map.shape[0]
    y_low = int(y1 + (y2 - y1) * 0.6)
    region = depth_map[y_low:y2, x1:x2]
    if region.size < 10:
        return None
    return np.median(region)

# =========================================================
# ⭐⭐ 문제 1, 문제 2 해결: 공식 DepthAnythingV2용 전처리 + infer 대체 함수 추가 ⭐⭐
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((518, 518)),  # DepthAnything 입력 규격
])

def depth_infer_official(model, frame, device):
    """DepthAnythingV2.forward() 기반 depth 추출"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(img_t)

    depth = depth.squeeze().cpu().numpy()

    # 원래 frame 크기로 resize
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    return depth

# =========================================================
# 6. Load Models
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("yolov8n.pt")

depth_model = DepthAnythingV2(encoder="vitl").to(device)
state = torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location=device)
depth_model.load_state_dict(state)
depth_model.eval()

CAR_DIMS = [1.5, 1.8, 4.0]   # H, W, L

# =========================================================
# 7. Main Loop
# =========================================================
cap = cv2.VideoCapture("/home/kjs/Downloads/test4.mp4")   # or your video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    yolos = yolo(frame)[0]

    # ⭐ 기존 infer(frame) → 공식 forward 기반 depth 추출로 대체
    depth_map = depth_infer_official(depth_model, frame, device)

    for i, box in enumerate(yolos.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # YOLO box 표시
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        median_depth = get_bbox_median_depth(depth_map, x1, y1, x2, y2)
        if median_depth is None:
            continue

        # 중심 pixel
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        X, Y, Z = pixel_to_camera(u, v, median_depth)

        yaw_raw = (u - cx) / fx
        yaw = smooth_yaw(i, yaw_raw)

        corners_3d = compute_box_3d(CAR_DIMS, [X, Y, Z], yaw)
        corners_2d = project_to_image(corners_3d, K)

        draw_3d_box(frame, corners_2d)

        cv2.putText(frame, f"Z={Z:.2f}m", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("3D Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

