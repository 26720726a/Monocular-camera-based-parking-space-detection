import cv2
import numpy as np


# =====================================================
# BEV Canvas + Lane
# =====================================================
def create_bev_canvas(w=1000, h=1000, scale=20):
    bev = np.zeros((h, w, 3), dtype=np.uint8)
    cx = w // 2

    # 중앙선
    cv2.line(bev, (cx, 0), (cx, h), (255, 255, 255), 2)

    # 차선 (5m 간격)
    lane_gap = int(5 * scale)

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
# Self Car (오른쪽 차선 중앙)
# =====================================================
def draw_self_car(bev, scale):
    h, w, _ = bev.shape
    center_x = w // 2

    lane_gap = 5 * scale
    cx = int(center_x + lane_gap / 2)
    cy = h - 10

    car_w = int(2.0 * scale)
    car_l = int(5.0 * scale)

    cv2.rectangle(
        bev,
        (cx - car_w // 2, cy - car_l),
        (cx + car_w // 2, cy),
        (0, 255, 0),
        -1
    )


# =====================================================
# Parking Slot Detection
# =====================================================
def find_parking_slots(boxes, min_gap=7.0, z_start=18.0):
    lanes = {"left": [], "right": []}

    for box in boxes:
        if box["X"] < 0:
            lanes["left"].append(box)
        else:
            lanes["right"].append(box)

    slots = []

    for lane, cars in lanes.items():
        cars = sorted(cars, key=lambda b: b["Z"])

        # 1️⃣ 차선 앞쪽
        if len(cars) > 0:
            first = cars[0]
            front_gap = (first["Z"] - first["L"] / 2) - z_start

            if front_gap >= min_gap:
                slots.append({
                    "lane": lane,
                    "Z_start": z_start,
                    "Z_end": first["Z"] - first["L"] / 2
                })

        # 2️⃣ 차량 사이
        for i in range(len(cars) - 1):
            c1 = cars[i]
            c2 = cars[i + 1]

            gap = (c2["Z"] - c2["L"] / 2) - (c1["Z"] + c1["L"] / 2)

            if gap >= min_gap:
                slots.append({
                    "lane": lane,
                    "Z_start": c1["Z"] + c1["L"] / 2,
                    "Z_end": c2["Z"] - c2["L"] / 2
                })

    return slots


# =====================================================
# Draw Parking Slots
# =====================================================
def draw_parking_slots(bev, slots, scale):
    h, w, _ = bev.shape
    center_x = w // 2

    lane_gap = 5 * scale
    offset_to_center = 2 * scale

    left_lane_x = int(center_x - 2 * lane_gap + offset_to_center)
    right_lane_x = int(center_x + 2 * lane_gap - offset_to_center)

    slot_width = int(2.5 * scale)

    for slot in slots:
        cx = left_lane_x if slot["lane"] == "left" else right_lane_x

        y1 = int(h - slot["Z_start"] * scale)
        y2 = int(h - slot["Z_end"] * scale)

        cv2.rectangle(
            bev,
            (cx - slot_width // 2, y2),
            (cx + slot_width // 2, y1),
            (0, 255, 0),
            3
        )


# =====================================================
# Main BEV Function
# =====================================================
def bev_from_detections(boxes, scale=20):
    if boxes is None:
        boxes = []

    bev = create_bev_canvas(w=1000, h=1000, scale=scale)
    h, w, _ = bev.shape
    center_x = w // 2

    lane_gap = 5 * scale
    offset_to_center = 2 * scale

    left_lane_center = center_x - 2 * lane_gap
    right_lane_center = center_x + 2 * lane_gap

    # 차량
    for box in boxes:
        X, Z, W, L = box["X"], box["Z"], box["W"], box["L"]

        if X < 0:
            lane_x = left_lane_center + offset_to_center
        else:
            lane_x = right_lane_center - offset_to_center

        cy = int(h - Z * scale)
        half_w = int(W * scale / 2)
        half_l = int(L * scale / 2)

        cv2.rectangle(
            bev,
            (int(lane_x - half_w), cy - half_l),
            (int(lane_x + half_w), cy + half_l),
            (70, 150, 150),
            -1
        )
        cv2.rectangle(
            bev,
            (int(lane_x - half_w), cy - half_l),
            (int(lane_x + half_w), cy + half_l),
            (0, 255, 255),
            2
        )

    # 주차 가능 공간
    parking_slots = find_parking_slots(boxes, min_gap=7.0)
    draw_parking_slots(bev, parking_slots, scale)

    # 자차
    draw_self_car(bev, scale)

    return bev

