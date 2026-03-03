"""Stereo geometry and depth estimation for ZED camera.

Shared by any modality that needs stereo depth (hand teleop, arm teleop, …).

Pinhole model recap
-------------------
Forward projection (3-D → pixel):
    u = fx * X/Z + cx
    v = fy * Y/Z + cy

Back-projection (pixel + depth → 3-D):
    X = (u - cx) * Z / fx      [m, left = negative]
    Y = (v - cy) * Z / fy      [m, up   = negative (camera Y points down)]

Stereo depth (disparity → Z):
    d = x_L - x_R              [raw pixels, principal point cancels]
    Z = fx * B / d             [metres]

Calibration
-----------
To get real values from the ZED SDK::

    import pyzed.sl as sl
    zed = sl.Camera()
    p = zed.get_camera_information().camera_configuration.calibration_parameters
    left = p.left_cam
    print(left.fx, left.fy, left.cx, left.cy)
    print(p.get_camera_baseline())   # metres

Port of Binocular-Teleop/vision/geometry.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PinholeCamera:
    """Intrinsic + stereo parameters for one ZED eye.

    All values assume the SBS frame has been split so each half is 1280×720.
    Replace with your unit's actual SDK values for best accuracy.
    """

    fx: float           # horizontal focal length [px]
    fy: float           # vertical   focal length [px]
    cx: float           # principal point x       [px]
    cy: float           # principal point y       [px]
    baseline_m: float   # left-right lens separation [m]


# ZED 2i factory defaults @ 720p SBS (each half = 1280×720).
# Swap with your SDK export for sub-percent accuracy.
ZED2I = PinholeCamera(
    fx         = 700.0,   # ZED 2i @ 720p ≈ 699.7
    fy         = 700.0,
    cx         = 638.0,   # ≈ 1280/2 − small offset
    cy         = 363.0,   # ≈ 720/2  − small offset
    baseline_m = 0.12,    # ZED 2i physical baseline = 12 cm
)

# Vertical pixel shift applied to the right frame in ZEDCamera.get_frames().
# Positive → right frame shifted DOWN (right lens sits higher in housing).
# Flip sign if debug shows ~30 px residual error after applying.
Y_OFFSET_PX: int = 30

# Anchor landmark indices used for depth averaging.
# Wrist + 3 MCP joints: stable, spread across palm, well-tracked.
_DEPTH_ANCHORS: List[int] = [0, 5, 9, 13]


# ---------------------------------------------------------------------------
# Epipolar constraint
# ---------------------------------------------------------------------------

def check_epipolar_constraint(
    y_left: float,
    y_right: float,
    tolerance_px: float = 5,
) -> tuple[bool, float]:
    """Return (is_valid, error_px).

    Y_OFFSET_PX is already baked into the frames by ZEDCamera, so this
    checks only residual misalignment.
    """
    error = abs(y_left - y_right)
    return error <= tolerance_px, error


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------

def back_project(
    u: float,
    v: float,
    z_m: float,
    cam: PinholeCamera = ZED2I,
) -> np.ndarray:
    """Convert pixel (u, v) + depth z_m [m] to 3-D camera-frame point.

    Returns
    -------
    np.ndarray shape (3,): [X_m, Y_m, Z_m]
        X_m: positive = right of camera
        Y_m: positive = below  camera  (camera Y points down)
        Z_m: positive = in front of camera
    """
    x_m = (u - cam.cx) * z_m / cam.fx
    y_m = (v - cam.cy) * z_m / cam.fy
    return np.array([x_m, y_m, z_m])


# ---------------------------------------------------------------------------
# Stereo depth
# ---------------------------------------------------------------------------

def stereo_depth_m(
    lm_l,
    lm_r,
    frame_w: int,
    cam: PinholeCamera = ZED2I,
    anchors: List[int] = _DEPTH_ANCHORS,
) -> float | None:
    """Robust depth estimate [m] by triangulating several stable landmarks.

    Averaging multiple landmarks rejects single-landmark outliers.
    The principal point cancels in the disparity formula (x_L − x_R).

    Parameters
    ----------
    lm_l, lm_r:
        MediaPipe landmark lists (21 entries each).
    frame_w:
        Pixel width of one half-frame (e.g. 1280).
    cam:
        PinholeCamera instance.
    anchors:
        Landmark indices to average over.

    Returns
    -------
    Median depth in metres, or None if every disparity was ≤ 0.
    """
    depths = []
    for i in anchors:
        x_l_px = lm_l[i].x * frame_w
        x_r_px = lm_r[i].x * frame_w
        disparity = x_l_px - x_r_px
        if disparity > 0.5:
            depths.append((cam.fx * cam.baseline_m) / disparity)
    return float(np.median(depths)) if depths else None


def stereo_hand_3d(
    lm_l,
    lm_r,
    frame_w: int,
    frame_h: int,
    cam: PinholeCamera = ZED2I,
    depth_min_m: float = 0.15,
    depth_max_m: float = 1.20,
) -> np.ndarray | None:
    """Return the 3-D wrist position [m] in the left-camera frame.

    Steps
    -----
    1. Triangulate depth from anchor landmarks (median → robust).
    2. Clamp depth to valid range.
    3. Back-project wrist pixel using pinhole model.

    Returns
    -------
    np.ndarray [X_m, Y_m, Z_m] or None if depth unavailable.
        X_m: positive = right   (mirror to get sim_x)
        Y_m: positive = down    (invert to get sim_z)
        Z_m: positive = forward (map to sim_y)
    """
    z_m = stereo_depth_m(lm_l, lm_r, frame_w, cam)
    if z_m is None:
        return None
    z_m = float(np.clip(z_m, depth_min_m, depth_max_m))

    u_wrist = lm_l[0].x * frame_w
    v_wrist = lm_l[0].y * frame_h
    return back_project(u_wrist, v_wrist, z_m, cam)


def triangulate_depth(
    x_left: float,
    x_right: float,
    cam: PinholeCamera = ZED2I,
) -> float | None:
    """Single-landmark depth [cm] via Z = fx * B / disparity.

    Prefer stereo_depth_m() — it averages multiple landmarks.
    """
    disparity = x_left - x_right
    if disparity <= 0:
        return None
    return (cam.fx * cam.baseline_m * 100.0) / disparity
