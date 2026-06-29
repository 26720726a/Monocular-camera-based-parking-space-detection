import cv2
import numpy as np

# =====================================================
# BEV Canvas + Lane
# =====================================================
def create_bev_canvas(w=1000, h=1000, scale=20):
    print(f"[BEV] Create canvas w={w}, h={h}, scale={scale}")

    bev = np.zeros((h, w, 3), dtype=np.uint8)
    cx = w // 2

    # 중앙선
    cv2.line(bev, (cx, 0), (cx, h), (255, 255, 255), 2)

    # 차선 (5m 간격)
    lane_gap = int(5 * scale)
    print(f"[BEV] Lane gap = {lane_gap}px (5m)")

    x = cx + lane_gap
    while x < w:
        cv2.line(bev, (x, 0), (x, h), (100, 100, 100), 1)
        x += lane_gap

    x = cx - lane_gap
    while x > 0:
        cv2.line(bev, (x, 0), (x, h), (100, 100, 100), 1)
        x -= lane_gap

    return bev


# =====================================================
# Self Car (고정 Ego Vehicle)
# =====================================================
def draw_self_car(bev, scale):
    h, w = bev.shape[:2]
    cx = w // 2
    y = int(h * 0.8)

    car_w = int(2.0 * scale)
    car_l = int(4.5 * scale)

    print(f"[BEV] Draw ego car at (x={cx}, y={y})")

    cv2.rectangle(
        bev,
        (cx - car_w // 2, y - car_l // 2),
        (cx + car_w // 2, y + car_l // 2),
        (0, 255, 0),
        -1
    )


# =====================================================
# BEV Projection (기준 차량 기준 상대 좌표)
# =====================================================
def bev_from_detections(detections, ref_depth, scale=20, canvas_size=(1000, 1000)):
    print(f"[BEV] Update BEV (ref_depth={ref_depth:.2f}m)")

    bev = create_bev_canvas(*canvas_size, scale)
    draw_self_car(bev, scale)

    h, w = canvas_size
    cx = w // 2
    base_y = int(h * 0.8)

    for i, det in enumerate(detections):
        X = det["X"]
        Z = det["Z"]
        label = det.get("label", "")
        color = det.get("color", (0, 255, 255))

        # 🔑 기준 차량 대비 상대 거리
        rel_Z = Z - ref_depth

        bev_x = int(cx + X * scale)
        bev_y = int(base_y - rel_Z * scale)

        print(
            f"[BEV][OBJ {i}] "
            f"X={X:.2f}m Z={Z:.2f}m → rel_Z={rel_Z:.2f}m | "
            f"BEV=({bev_x},{bev_y})"
        )

        cv2.circle(bev, (bev_x, bev_y), 6, color, -1)
        cv2.putText(
            bev,
            f"{label} {rel_Z:.1f}m",
            (bev_x + 5, bev_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    return bev

