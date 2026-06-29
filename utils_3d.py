import numpy as np
import cv2


# -------------------------------------------------------
# 정확한 KITTI 스타일 3D Box Corner 계산
# -------------------------------------------------------
def compute_3d_box_corners(x, y, z, w, h, l, ry):
    """
    x,y,z  : 3D center (camera coords)
    w,h,l  : object size
    ry     : yaw rotation
    """

    # rotation matrix (yaw)
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1,         0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # KITTI order (height goes down)
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_corners = [ 0,   -h,   -h,    0,     0,   -h,   -h,     0 ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]

    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = R @ corners
    corners += np.array([[x], [y], [z]])

    return corners


# -------------------------------------------------------
# Camera → Image projection
# -------------------------------------------------------
def project_corners_to_image(corners, P2):
    pts_homo = np.vstack([corners, np.ones((1, 8))])
    pts_2d = P2 @ pts_homo
    pts_2d /= pts_2d[2]
    return pts_2d[:2]


# -------------------------------------------------------
# draw 3D box
# -------------------------------------------------------
def draw_3d_box(img, corners2d, color=(255,255,0)):
    pts = corners2d.T.astype(int)

    # front
    for i in range(4):
        cv2.line(img, pts[i], pts[(i+1) % 4], color, 2)

    # back
    for i in range(4, 8):
        cv2.line(img, pts[i], pts[4 + (i+1) % 4], color, 2)

    # connection
    for i in range(4):
        cv2.line(img, pts[i], pts[i+4], color, 2)

    return img

