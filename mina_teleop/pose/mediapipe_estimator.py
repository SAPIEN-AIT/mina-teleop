"""MediaPipe Pose wrapper for single-arm landmark extraction.

Runs camera capture and pose estimation in a background daemon thread.
The main thread (sim loop) calls get_landmarks() non-blocking to read
the latest available result.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe landmark indices used for arm estimation
# ---------------------------------------------------------------------------

_POSE = mp.solutions.pose.PoseLandmark

_LANDMARK_IDS = {
    "right": {
        "shoulder":          _POSE.RIGHT_SHOULDER,
        "elbow":             _POSE.RIGHT_ELBOW,
        "wrist":             _POSE.RIGHT_WRIST,
        "opposite_shoulder": _POSE.LEFT_SHOULDER,
    },
    "left": {
        "shoulder":          _POSE.LEFT_SHOULDER,
        "elbow":             _POSE.LEFT_ELBOW,
        "wrist":             _POSE.LEFT_WRIST,
        "opposite_shoulder": _POSE.RIGHT_SHOULDER,
    },
}

_MIN_VISIBILITY = 0.5  # discard landmarks below this confidence


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ArmLandmarks:
    """3D landmark positions for one arm extracted from MediaPipe Pose.

    Coordinates are in MediaPipe's normalised camera frame:
      x: [0, 1]  left → right of image
      y: [0, 1]  top  → bottom of image
      z: depth relative to hip midpoint (negative = closer to camera)
    """
    shoulder:          np.ndarray   # (3,)
    elbow:             np.ndarray   # (3,)
    wrist:             np.ndarray   # (3,)
    opposite_shoulder: np.ndarray   # (3,) — used as torso reference
    visibility:        float        # minimum visibility across the three joints


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class MediaPipeArmEstimator:
    """Runs MediaPipe Pose in a background thread and exposes the latest arm
    landmarks via a non-blocking getter.

    Parameters
    ----------
    side:
        ``'right'`` or ``'left'`` — which arm to extract.
    camera_id:
        OpenCV camera index (0 = default built-in webcam on Mac).
    min_detection_confidence:
        Passed to MediaPipe Pose.
    min_tracking_confidence:
        Passed to MediaPipe Pose.
    """

    def __init__(
        self,
        side: str = "right",
        camera_id: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        self.side = side
        self.camera_id = camera_id
        self._detection_conf = min_detection_confidence
        self._tracking_conf = min_tracking_confidence

        self._lock = threading.Lock()
        self._landmarks: ArmLandmarks | None = None
        self._debug_frame: np.ndarray | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background camera + pose estimation thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_landmarks(self) -> ArmLandmarks | None:
        """Return the most recent arm landmarks, or ``None`` if not yet ready."""
        with self._lock:
            return self._landmarks

    def get_debug_frame(self) -> np.ndarray | None:
        """Return the most recent annotated BGR frame, or ``None``."""
        with self._lock:
            return self._debug_frame

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        ids = _LANDMARK_IDS[self.side]
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(
            min_detection_confidence=self._detection_conf,
            min_tracking_confidence=self._tracking_conf,
        ) as pose:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

                # MediaPipe requires RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True

                annotated = frame.copy()

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    lm = results.pose_world_landmarks.landmark  # 3D world coords

                    def _get(key: str) -> tuple[np.ndarray, float]:
                        idx = ids[key]
                        l = lm[idx]
                        return np.array([l.x, l.y, l.z], dtype=np.float32), l.visibility

                    shoulder,          vis_s = _get("shoulder")
                    elbow,             vis_e = _get("elbow")
                    wrist,             vis_w = _get("wrist")
                    opposite_shoulder, _     = _get("opposite_shoulder")

                    min_vis = min(vis_s, vis_e, vis_w)

                    if min_vis >= _MIN_VISIBILITY:
                        arm_lm = ArmLandmarks(
                            shoulder=shoulder,
                            elbow=elbow,
                            wrist=wrist,
                            opposite_shoulder=opposite_shoulder,
                            visibility=float(min_vis),
                        )
                        with self._lock:
                            self._landmarks = arm_lm

                with self._lock:
                    self._debug_frame = annotated

        cap.release()
