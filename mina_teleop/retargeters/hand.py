"""LEAP Hand retargeter.

Converts 21 MediaPipe hand landmarks into 16 LEAP joint angles via
direct angle mapping — no IK required for fingers.

Ported from Binocular-Teleop/robots/leap_hand/ik_retargeting.py
(Edgard, SAPIEN-AIT).

LEAP joint ordering (16 DOF, right hand):
    Index  Name
    -----  -----
    0      index_mcp     (flexion)
    1      index_pip
    2      index_dip
    3      index_abd     (abduction/spread)
    4      middle_mcp
    5      middle_pip
    6      middle_dip
    7      middle_abd
    8      ring_mcp
    9      ring_pip
    10     ring_dip
    11     ring_abd
    12     thumb_cmc
    13     thumb_axial
    14     thumb_mcp
    15     thumb_ip

NOTE: Verify this ordering against the actual leap_hand XML before
running on hardware. The ordering in the MJCF actuator list is
authoritative.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# ---------------------------------------------------------------------------
# Calibration: max bend angles observed during full closure (radians).
# Derived from MediaPipe model_complexity=1 measurements.
# ---------------------------------------------------------------------------
_MP_MAX_BEND   = 1.8    # rad — max flex angle in MediaPipe space
_LEAP_MAX_FLEX = 1.57   # rad — ~90° LEAP joint limit for MCP/PIP/DIP

_EPS = 1e-6


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two 3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _flex_angle(lm: np.ndarray, proximal: int, middle: int, distal: int) -> float:
    """Flexion angle at a joint defined by three consecutive landmarks."""
    v1 = lm[proximal] - lm[middle]
    v2 = lm[distal]   - lm[middle]
    raw = _angle_between(v1, v2)
    # Convert: 180° (straight) → 0 flex, 0° (fully bent) → max flex
    return float(np.pi - raw)


def _spread_angle(
    lm: np.ndarray,
    ref_mcp: int,
    finger_mcp: int,
) -> float:
    """Lateral spread (abduction) angle between two MCP joints.

    Scale-invariant: normalized against wrist-to-middle-MCP distance.
    """
    wrist_to_mid = lm[MIDDLE_MCP] - lm[WRIST]
    scale = np.linalg.norm(wrist_to_mid) + _EPS
    lateral = lm[finger_mcp] - lm[ref_mcp]
    return float(np.linalg.norm(lateral) / scale)


class LeapRetargeter:
    """Maps MediaPipe 21 hand landmarks to 16 LEAP joint angles.

    Design principles:
    - Scale invariant: works regardless of hand distance from camera.
    - Direct angle mapping: no IK solver, purely geometric.
    - Thumb handled separately with opposition tracking.

    Parameters
    ----------
    scale:
        Global output scale [0, 1]. Reduce if the robot motion feels
        too aggressive relative to the human hand.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def retarget(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert landmarks to LEAP joint targets.

        Parameters
        ----------
        landmarks:
            Shape ``(21, 3)`` — MediaPipe world landmarks.

        Returns
        -------
        np.ndarray
            Shape ``(16,)`` — joint angles in radians, ordered per
            the LEAP joint table in this module's docstring.
        """
        lm = landmarks
        targets = np.zeros(16)
        s = self.scale

        # ── Index finger ─────────────────────────────────────────────
        targets[0] = s * self._flex_scaled(lm, WRIST,     INDEX_MCP, INDEX_PIP)
        targets[1] = s * self._flex_scaled(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP)
        targets[2] = s * self._flex_scaled(lm, INDEX_PIP, INDEX_DIP, INDEX_TIP)
        targets[3] = s * _spread_angle(lm, MIDDLE_MCP, INDEX_MCP)

        # ── Middle finger ─────────────────────────────────────────────
        targets[4] = s * self._flex_scaled(lm, WRIST,      MIDDLE_MCP, MIDDLE_PIP)
        targets[5] = s * self._flex_scaled(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP)
        targets[6] = s * self._flex_scaled(lm, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
        targets[7] = 0.0  # middle finger — no abduction DOF on LEAP

        # ── Ring finger ───────────────────────────────────────────────
        targets[8]  = s * self._flex_scaled(lm, WRIST,    RING_MCP, RING_PIP)
        targets[9]  = s * self._flex_scaled(lm, RING_MCP, RING_PIP, RING_DIP)
        targets[10] = s * self._flex_scaled(lm, RING_PIP, RING_DIP, RING_TIP)
        targets[11] = s * _spread_angle(lm, MIDDLE_MCP, RING_MCP)

        # ── Thumb ─────────────────────────────────────────────────────
        targets[12] = s * self._flex_scaled(lm, WRIST,     THUMB_CMC, THUMB_MCP)  # CMC flex
        targets[13] = s * self._thumb_axial(lm)                                    # axial rotation
        targets[14] = s * self._flex_scaled(lm, THUMB_CMC, THUMB_MCP, THUMB_IP)   # MCP flex
        targets[15] = s * self._flex_scaled(lm, THUMB_MCP, THUMB_IP, THUMB_TIP)   # IP flex

        return targets

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flex_scaled(
        lm: np.ndarray,
        proximal: int,
        joint: int,
        distal: int,
    ) -> float:
        """Flexion angle scaled from MediaPipe range to LEAP range."""
        raw = _flex_angle(lm, proximal, joint, distal)
        scaled = (raw / _MP_MAX_BEND) * _LEAP_MAX_FLEX
        return float(np.clip(scaled, 0.0, _LEAP_MAX_FLEX))

    @staticmethod
    def _thumb_axial(lm: np.ndarray) -> float:
        """Thumb axial (opposition) angle.

        Approximated from the angle between the thumb CMC-MCP vector
        and the palm plane normal.
        """
        palm_x = lm[INDEX_MCP]  - lm[PINKY_MCP]
        palm_y = lm[MIDDLE_MCP] - lm[WRIST]
        palm_normal = np.cross(palm_x, palm_y)
        thumb_dir   = lm[THUMB_MCP] - lm[THUMB_CMC]
        angle = _angle_between(palm_normal, thumb_dir)
        # Map [0, π] → [0, 1.0] rad opposition range
        return float(np.clip(angle / np.pi, 0.0, 1.0))
