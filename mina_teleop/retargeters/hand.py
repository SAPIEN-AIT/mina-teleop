"""Direct angle-based retargeter: MediaPipe 3-D landmarks → LEAP joint angles.

Why direct angle mapping (not Jacobian IK)?
  Without full wrist orientation tracking, we can't correctly map a
  MediaPipe image-plane vector to a 3-D world-space fingertip target.
  Direct angle mapping avoids the problem entirely:
    1. Compute the bend angle at each MediaPipe joint from its two adjacent
       bone vectors (using all three x/y/z coordinates — depth is captured).
    2. Scale the angle into the corresponding LEAP joint range.

LEAP actuator ordering (16 DOF, right hand) — matches right_hand.xml:
    Index  Name
    -----  --------
    0      if_mcp    index  metacarpophalangeal  flexion
    1      if_rot    index  rotational           abduction
    2      if_pip    index  proximal IP          flexion
    3      if_dip    index  distal IP            flexion
    4      mf_mcp    middle MCP flexion
    5      mf_rot    middle rotational abduction
    6      mf_pip    middle PIP flexion
    7      mf_dip    middle DIP flexion
    8      rf_mcp    ring   MCP flexion
    9      rf_rot    ring   rotational abduction
    10     rf_pip    ring   PIP flexion
    11     rf_dip    ring   DIP flexion
    12     th_cmc    thumb  carpometacarpal
    13     th_axl    thumb  axial (opposition)
    14     th_mcp    thumb  metacarpophalangeal
    15     th_ipl    thumb  interphalangeal

MediaPipe landmark indices (21 total):
    WRIST=0
    THUMB : CMC=1  MCP=2  IP=3  TIP=4
    INDEX : MCP=5  PIP=6  DIP=7  TIP=8
    MIDDLE: MCP=9  PIP=10 DIP=11 TIP=12
    RING  : MCP=13 PIP=14 DIP=15 TIP=16

Port of Binocular-Teleop/robots/leap_hand/ik_retargeting.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------
# Set True to print the running-max bend angle seen at each joint.
# Run with a tight fist — the peak value is your real _MP_MAX.
DEBUG_ANGLES: bool = False

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP_LM = 2;  THUMB_IP = 3;  THUMB_TIP = 4
INDEX_MCP    = 5;  INDEX_PIP    = 6;  INDEX_DIP = 7;  INDEX_TIP  = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP   = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP     = 13; RING_PIP     = 14; RING_DIP   = 15; RING_TIP   = 16

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Maximum inter-bone angle MediaPipe produces at a fully-curled joint.
# With model_complexity=1 the observable max is ~1.8 rad (~103°).
_MP_MAX  = 1.8   # radians

# Practical LEAP upper limits (slightly below XML max to stay away from hard stops)
_MCP_MAX  = 2.0   # LEAP xml max 2.23
_PIP_MAX  = 1.7   # LEAP xml max 1.885
_DIP_MAX  = 1.8   # LEAP xml max 2.042
_ROT_LIM  = 0.6   # abduction clamp (xml ±1.047)
_CMC_MAX  = 1.8   # th_cmc
_AXL_MAX  = 1.8   # th_axl
_TMCP_MAX = 2.0   # th_mcp
_IPL_MAX  = 1.5   # th_ipl


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _lm3(lm, i: int) -> np.ndarray:
    """Landmark list → (x, y, z) array."""
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two bone vectors at a joint.

    0   → straight finger
    π/2 → 90° bend
    π   → fully curled (theoretical)

    All three MediaPipe coordinates (x, y, z) are used, so finger depth
    contributes correctly.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _scale(angle: float, leap_max: float) -> float:
    """Map MediaPipe bend ∈ [0, _MP_MAX] → LEAP angle ∈ [0, leap_max]."""
    return float(np.clip(angle / _MP_MAX * leap_max, 0.0, leap_max))


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion (w, x, y, z) — MuJoCo convention.

    Numerically stable Shepperd method.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,                 (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# ---------------------------------------------------------------------------
# Palm orientation helper (module-level, used by teleop script for wrist)
# ---------------------------------------------------------------------------

def palm_quat(lm) -> np.ndarray:
    """Compute LEAP palm orientation as MuJoCo quaternion (w, x, y, z).

    Palm frame (MediaPipe camera space):
      long_axis : WRIST → MIDDLE_MCP          (finger direction)
      lat_axis  : INDEX_MCP → RING_MCP         (across knuckles)
      normal    : cross(lat_axis, long_axis)   (palm-outward normal)

    Camera → MuJoCo frame:
      cam X (right)   → muj X (right)
      cam Z (forward) → muj Y (forward)
      cam Y (down)    → −muj Z (up)
    """
    w_lm = _lm3(lm, WRIST)
    imcp = _lm3(lm, INDEX_MCP)
    mmcp = _lm3(lm, MIDDLE_MCP)
    rmcp = _lm3(lm, RING_MCP)

    long_v = mmcp - w_lm
    lat_v  = rmcp - imcp
    norm_v = np.cross(lat_v, long_v)

    # Gram-Schmidt orthonormalisation
    long_v = long_v / (np.linalg.norm(long_v) + 1e-6)
    norm_v = norm_v / (np.linalg.norm(norm_v) + 1e-6)
    lat_v  = np.cross(long_v, norm_v)
    lat_v  = lat_v  / (np.linalg.norm(lat_v)  + 1e-6)

    R_cam = np.column_stack([lat_v, long_v, norm_v])

    # Camera → MuJoCo change-of-basis
    T = np.array([[1,  0,  0],
                  [0,  0,  1],
                  [0, -1,  0]], dtype=float)
    R_muj = T @ R_cam
    return _rot_to_quat(R_muj)


# ---------------------------------------------------------------------------
# Main retargeter
# ---------------------------------------------------------------------------

class IKRetargeter:
    """Maps MediaPipe hand landmarks → 16 LEAP joint angles.

    Purely angle-based — no IK solver, no world-frame information required.

    Parameters
    ----------
    model:
        Optional MuJoCo model (kept for API compatibility; used to track
        running-max angles when DEBUG_ANGLES is True).

    Usage
    -----
        ik = IKRetargeter(model)          # or IKRetargeter()
        q  = ik.retarget(landmarks)       # shape (16,)
    """

    def __init__(self, model=None, n_iters: int = 15, step: float = 0.5) -> None:
        self.model = model
        # Running-max angles per bend joint (used by DEBUG_ANGLES).
        # [if_mcp, if_pip, if_dip, mf_mcp, mf_pip, mf_dip,
        #  rf_mcp, rf_pip, rf_dip, th_cmc, th_mcp, th_ipl]
        self._angle_max = np.zeros(12)

    def retarget(self, lm) -> np.ndarray:
        """Compute 16 LEAP joint angles from a MediaPipe landmark list.

        Parameters
        ----------
        lm:
            ``result.multi_hand_landmarks[0].landmark``  (21 entries).

        Returns
        -------
        np.ndarray shape (16,):
            [if_mcp, if_rot, if_pip, if_dip,
             mf_mcp, mf_rot, mf_pip, mf_dip,
             rf_mcp, rf_rot, rf_pip, rf_dip,
             th_cmc, th_axl, th_mcp, th_ipl]
        """
        q = np.zeros(16)

        w  = _lm3(lm, WRIST)
        mm = _lm3(lm, MIDDLE_MCP)

        # ── Index finger (0-3: mcp, rot, pip, dip) ───────────────────────
        im  = _lm3(lm, INDEX_MCP);  ip  = _lm3(lm, INDEX_PIP)
        idd = _lm3(lm, INDEX_DIP);  it  = _lm3(lm, INDEX_TIP)

        q[0] = _scale(_bend(im - w,  ip  - im), _MCP_MAX)
        hand_scale = float(np.linalg.norm((mm - w)[:2])) + 1e-6
        q[1] = float(np.clip((im[0] - mm[0]) / hand_scale * 1.5,
                              -_ROT_LIM, _ROT_LIM))
        q[2] = _scale(_bend(ip - im,  idd - ip),  _PIP_MAX)
        q[3] = _scale(_bend(idd - ip, it  - idd), _DIP_MAX)

        # ── Middle finger (4-7) ───────────────────────────────────────────
        mmp = _lm3(lm, MIDDLE_MCP); mpi = _lm3(lm, MIDDLE_PIP)
        mdi = _lm3(lm, MIDDLE_DIP); mti = _lm3(lm, MIDDLE_TIP)

        q[4] = _scale(_bend(mmp - w,   mpi - mmp), _MCP_MAX)
        mid_ref_x = (_lm3(lm, INDEX_MCP)[0] + _lm3(lm, RING_MCP)[0]) * 0.5
        q[5] = float(np.clip((mid_ref_x - mm[0]) / hand_scale * 1.0,
                              -_ROT_LIM, _ROT_LIM))
        q[6] = _scale(_bend(mpi - mmp, mdi - mpi), _PIP_MAX)
        q[7] = _scale(_bend(mdi - mpi, mti - mdi), _DIP_MAX)

        # ── Ring finger (8-11) ────────────────────────────────────────────
        rm  = _lm3(lm, RING_MCP);  rp  = _lm3(lm, RING_PIP)
        rd  = _lm3(lm, RING_DIP);  rt  = _lm3(lm, RING_TIP)

        q[8]  = _scale(_bend(rm - w,  rp - rm), _MCP_MAX)
        q[9]  = float(np.clip((mm[0] - rm[0]) / hand_scale * 1.5,
                               -_ROT_LIM, _ROT_LIM))
        q[10] = _scale(_bend(rp - rm, rd - rp), _PIP_MAX)
        q[11] = _scale(_bend(rd - rp, rt - rd), _DIP_MAX)

        # ── Thumb (12-15: th_cmc, th_axl, th_mcp, th_ipl) ────────────────
        tc  = _lm3(lm, THUMB_CMC)
        tm  = _lm3(lm, THUMB_MCP_LM)
        tip = _lm3(lm, THUMB_IP)
        tt  = _lm3(lm, THUMB_TIP)

        q[12] = _scale(_bend(tc - w,  tm  - tc),  _CMC_MAX)
        opposition = float(np.clip(
            (_lm3(lm, INDEX_MCP)[0] - tt[0]) / hand_scale * 1.2, 0.0, 1.0))
        q[13] = opposition * _AXL_MAX
        q[14] = _scale(_bend(tm - tc,  tip - tm), _TMCP_MAX)
        q[15] = _scale(_bend(tip - tm, tt  - tip), _IPL_MAX)

        if DEBUG_ANGLES:
            raw = np.array([
                _bend(im - w,   ip  - im),
                _bend(ip - im,  idd - ip),
                _bend(idd - ip, it  - idd),
                _bend(mmp - w,  mpi - mmp),
                _bend(mpi - mmp, mdi - mpi),
                _bend(mdi - mpi, mti - mdi),
                _bend(rm - w,   rp  - rm),
                _bend(rp - rm,  rd  - rp),
                _bend(rd - rp,  rt  - rd),
                _bend(tc - w,   tm  - tc),
                _bend(tm - tc,  tip - tm),
                _bend(tip - tm, tt  - tip),
            ])
            self._angle_max = np.maximum(self._angle_max, raw)
            names = ['if_mcp', 'if_pip', 'if_dip',
                     'mf_mcp', 'mf_pip', 'mf_dip',
                     'rf_mcp', 'rf_pip', 'rf_dip',
                     'th_cmc', 'th_mcp', 'th_ipl']
            parts = [f"{n}={v:.2f}(max={m:.2f})"
                     for n, v, m in zip(names, raw, self._angle_max)]
            print("ANGLES  " + "  ".join(parts))

        return q
