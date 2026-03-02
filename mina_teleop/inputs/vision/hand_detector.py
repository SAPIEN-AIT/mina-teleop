"""MediaPipe Hands detector for stereo hand tracking.

Runs two independent MediaPipe Hands instances — one per camera —
to extract 21 3D hand landmarks from each view.

Ported from Binocular-Teleop/vision/detectors.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices (21 per hand)
# ---------------------------------------------------------------------------
WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

N_HAND_LANDMARKS = 21


@dataclass
class HandLandmarks:
    """21 MediaPipe hand landmarks.

    Parameters
    ----------
    points:
        Shape ``(21, 3)`` — (x, y, z) world coordinates, metric scale.
        MediaPipe Hands world landmarks are rooted at the wrist.
    visibility:
        Shape ``(21,)`` — per-landmark presence score [0, 1].
    """
    points: np.ndarray      # (21, 3)
    visibility: np.ndarray  # (21,)

    @property
    def valid(self) -> bool:
        """True when average visibility is above threshold."""
        return float(self.visibility.mean()) > 0.3


@dataclass
class StereoHandObservation:
    """Landmarks from both camera views for one hand."""
    left_cam:  HandLandmarks | None
    right_cam: HandLandmarks | None

    @property
    def valid(self) -> bool:
        return (self.left_cam is not None and self.left_cam.valid
                and self.right_cam is not None and self.right_cam.valid)


class StereoHandDetector:
    """Runs two MediaPipe Hands instances in parallel.

    One instance per camera (left, right). Optimized for speed
    (model_complexity=0).

    Parameters
    ----------
    max_hands:
        Maximum number of hands to detect per frame (1 or 2).
    min_detection_confidence:
        Detection confidence threshold.
    min_tracking_confidence:
        Tracking confidence threshold.
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        mp_hands = mp.solutions.hands
        self._tracker_left = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._tracker_right = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils

    def process(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> StereoHandObservation:
        """Run detection on a stereo frame pair.

        Parameters
        ----------
        left_bgr, right_bgr:
            BGR frames from the left and right cameras (H, W, 3).

        Returns
        -------
        StereoHandObservation
            Landmark arrays from both views, or None if not detected.
        """
        left_lm  = self._run(self._tracker_left,  left_bgr)
        right_lm = self._run(self._tracker_right, right_bgr)
        return StereoHandObservation(left_cam=left_lm, right_cam=right_lm)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: any,
    ) -> None:
        """Draw MediaPipe skeleton overlay on a frame (in-place)."""
        if result and result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._tracker_left.close()
        self._tracker_right.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(tracker, bgr: np.ndarray) -> HandLandmarks | None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = tracker.process(rgb)
        if not result.multi_hand_world_landmarks:
            return None
        # Take the first detected hand
        wlm = result.multi_hand_world_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in wlm.landmark])
        # MediaPipe Hands world landmarks don't have visibility — use presence
        if result.multi_hand_landmarks:
            vis = np.array([lm.visibility for lm in result.multi_hand_landmarks[0].landmark])
        else:
            vis = np.ones(N_HAND_LANDMARKS)
        return HandLandmarks(points=points, visibility=vis)
