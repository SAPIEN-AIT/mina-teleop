"""Body frame normalization for MediaPipe landmarks.

Expresses all landmarks relative to a torso reference frame built from
the shoulder midpoint. This removes dependency on camera placement and
operator position/orientation relative to the camera.

Torso frame definition
----------------------
  Origin : midpoint of left and right shoulders
  X axis : left_shoulder → right_shoulder  (lateral, points right)
  Y axis : orthogonalized world-up vs X     (vertical, points up)
  Z axis : X × Y                            (forward, points toward camera)
"""

from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12

_EPS = 1e-6


class BodyFrameNormalizer:
    """Transforms a (33, 3) MediaPipe landmark array into the torso frame.

    The torso frame is recomputed every call from the shoulder landmarks,
    so it tracks the operator's body orientation in real time.

    Parameters
    ----------
    world_up:
        The up direction in the camera/world coordinate system.
        MediaPipe world landmarks use Y-up, so the default is correct.
    """

    def __init__(self, world_up: np.ndarray | None = None) -> None:
        self._world_up = (
            np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if world_up is None
            else np.asarray(world_up, dtype=np.float64)
        )

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """Transform landmarks into the torso reference frame.

        Parameters
        ----------
        landmarks:
            Shape ``(33, 3)`` — raw MediaPipe world landmarks in camera frame.

        Returns
        -------
        np.ndarray
            Shape ``(33, 3)`` — landmarks expressed in the torso frame.
            The shoulder midpoint is the origin; axes follow the torso.
        """
        assert landmarks.shape == (33, 3), (
            f"Expected shape (33, 3), got {landmarks.shape}"
        )

        origin, R = self._compute_frame(landmarks)

        # Translate then rotate all landmarks into the torso frame
        translated = landmarks - origin          # (33, 3)
        return (R @ translated.T).T             # (33, 3)

    def get_frame(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the torso frame origin and rotation matrix.

        Returns
        -------
        origin: np.ndarray shape (3,)
        R: np.ndarray shape (3, 3)  — rows are X, Y, Z axes of torso frame
        """
        return self._compute_frame(landmarks)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_frame(
        self, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        l_shoulder = landmarks[_LEFT_SHOULDER].astype(np.float64)
        r_shoulder = landmarks[_RIGHT_SHOULDER].astype(np.float64)

        origin = 0.5 * (l_shoulder + r_shoulder)

        # X axis — lateral (left → right)
        x = r_shoulder - l_shoulder
        x = x / (np.linalg.norm(x) + _EPS)

        # Y axis — orthogonalized vertical (Gram-Schmidt)
        up = self._world_up
        y = up - np.dot(up, x) * x
        y = y / (np.linalg.norm(y) + _EPS)

        # Z axis — forward (right-hand rule)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + _EPS)

        # Rotation matrix: each row is one axis of the torso frame
        R = np.stack([x, y, z], axis=0)  # (3, 3)

        return origin, R
