"""Converts MediaPipe arm landmarks to robot joint angles.

Maps 3D landmark geometry to 5 joint angles per arm:

    shoulder_pitch  — forward/backward  (uses MediaPipe world-space z depth)
    shoulder_roll   — elevation from side (uses world-space y)
    shoulder_yaw    — upper-arm twist    (not recoverable monocular → 0)
    elbow_pitch     — elbow bend         (angle between upper-arm and forearm)
    elbow_roll      — forearm twist      (not recoverable monocular → 0)

A calibration step captures a neutral pose and zeroes the output at that pose.
An exponential moving average smooths the output to reduce jitter.
"""

from __future__ import annotations

import numpy as np

from mina_teleop.inputs.vision.mediapipe_engine import ArmLandmarks, BimanualArmLandmarks

_EPS = 1e-6


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + _EPS)


def _safe_acos(x: float) -> float:
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


class ArmRetargeter:
    """Converts ArmLandmarks to a 5-DOF joint angle vector.

    Parameters
    ----------
    side:
        ``'right'`` or ``'left'``.
    smoothing:
        EMA coefficient for the *previous* frame (0 = no smoothing, 0.9 = heavy).
        A value of 0.7 keeps 70 % of the previous angle and 30 % of the new one.
    """

    NUM_JOINTS = 5  # per arm: pitch, roll, yaw, elbow_pitch, elbow_roll

    def __init__(self, side: str = "right", smoothing: float = 0.7) -> None:
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        self.side = side
        self.smoothing = smoothing

        self._neutral: np.ndarray | None = None           # calibrated offset
        self._prev: np.ndarray = np.zeros(self.NUM_JOINTS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calibrate(self, landmarks: ArmLandmarks) -> None:
        """Capture the current pose as the neutral (zero) position.

        Call this while the operator holds a comfortable rest pose with arms
        at their sides. All subsequent ``retarget()`` outputs will be relative
        to this pose.
        """
        self._neutral = self._compute_raw(landmarks)
        self._prev[:] = 0.0
        print(f"[ArmRetargeter/{self.side}] Calibrated. Neutral angles (rad): "
              f"{np.round(self._neutral, 3)}")

    def retarget(self, landmarks: ArmLandmarks) -> np.ndarray:
        """Return smoothed 5-DOF joint angles in radians.

        Returns
        -------
        np.ndarray
            Shape ``(5,)`` — [shoulder_pitch, shoulder_roll, shoulder_yaw,
            elbow_pitch, elbow_roll].
            shoulder_yaw and elbow_roll are always 0 (not recoverable monoculary).
        """
        raw = self._compute_raw(landmarks)

        if self._neutral is not None:
            raw -= self._neutral

        # Exponential moving average
        smoothed = self.smoothing * self._prev + (1.0 - self.smoothing) * raw
        self._prev[:] = smoothed
        return smoothed.copy()

    # ------------------------------------------------------------------
    # Angle computation
    # ------------------------------------------------------------------

    def _compute_raw(self, lm: ArmLandmarks) -> np.ndarray:
        """Compute raw (un-calibrated, un-smoothed) joint angles."""
        upper_arm = _norm(lm.elbow - lm.shoulder)
        forearm   = _norm(lm.wrist - lm.elbow)

        shoulder_pitch = self._shoulder_pitch(upper_arm)
        shoulder_roll  = self._shoulder_roll(upper_arm)
        shoulder_yaw   = 0.0
        elbow_pitch    = self._elbow_pitch(upper_arm, forearm)
        elbow_roll     = 0.0

        return np.array(
            [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll],
            dtype=np.float32,
        )

    def _shoulder_pitch(self, upper_arm: np.ndarray) -> float:
        """Forward/backward rotation of the upper arm.

        Uses the depth (z) component from MediaPipe world landmarks.
        MediaPipe world coords: x = right, y = up, z = toward camera.
        Positive pitch = arm raised forward.
        """
        # Project onto the sagittal plane (y-z), compute elevation from vertical
        return float(np.arctan2(-upper_arm[2], -upper_arm[1] + _EPS))

    def _shoulder_roll(self, upper_arm: np.ndarray) -> float:
        """Side elevation of the upper arm (arm raising from rest position).

        Positive roll = arm raised away from body.
        """
        # In MediaPipe world coords y points up; arm hanging down → upper_arm.y < 0
        # Elevation from the downward vertical direction
        return float(np.arctan2(upper_arm[0], -upper_arm[1] + _EPS))

    def _elbow_pitch(self, upper_arm: np.ndarray, forearm: np.ndarray) -> float:
        """Elbow bend angle.

        0 = arm fully extended, increases as elbow bends.
        """
        # Angle between upper-arm and forearm vectors, measured from straight
        cos_angle = float(np.dot(-upper_arm, forearm))
        return float(np.pi) - _safe_acos(cos_angle)


class BimanualRetargeter:
    """Retargets both arms simultaneously, returning a full 10-DOF joint vector.

    Internally wraps two ``ArmRetargeter`` instances (one per side).
    Joint ordering matches ``ArmSim``:
        [0:5]  left  arm — shoulder_pitch, roll, yaw, elbow_pitch, elbow_roll
        [5:10] right arm — shoulder_pitch, roll, yaw, elbow_pitch, elbow_roll

    Parameters
    ----------
    smoothing:
        EMA smoothing applied to both arms.
    """

    NUM_JOINTS = 10

    def __init__(self, smoothing: float = 0.7) -> None:
        self._left  = ArmRetargeter(side="left",  smoothing=smoothing)
        self._right = ArmRetargeter(side="right", smoothing=smoothing)

    def calibrate(self, landmarks: BimanualArmLandmarks) -> None:
        """Capture the current pose as neutral for both arms."""
        self._left.calibrate(landmarks.left)
        self._right.calibrate(landmarks.right)

    def retarget(self, landmarks: BimanualArmLandmarks) -> np.ndarray:
        """Return smoothed 10-DOF joint angles in radians.

        Returns
        -------
        np.ndarray
            Shape ``(10,)`` — left arm (0:5) then right arm (5:10).
        """
        targets = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        targets[:5]  = self._left.retarget(landmarks.left)
        targets[5:]  = self._right.retarget(landmarks.right)
        return targets
