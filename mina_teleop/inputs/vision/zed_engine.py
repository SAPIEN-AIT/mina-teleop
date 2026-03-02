"""ZED stereo camera interface.

Wraps a ZED camera (ZED Mini, ZED 2, ZED 2i) via OpenCV capture.
The ZED appears as a single wide-format device that returns a
side-by-side stereo frame — this class splits it into left/right.

Ported from Binocular-Teleop/vision/camera.py (Edgard, SAPIEN-AIT).
"""

from __future__ import annotations

import cv2
import numpy as np


class ZEDCamera:
    """Interface for a ZED stereo camera via OpenCV.

    The ZED SDK is not required — the camera is accessed as a standard
    V4L2/UVC device. The output is a 1280×720 side-by-side frame that
    is split into left (640×720) and right (640×720) images.

    Parameters
    ----------
    camera_id:
        OpenCV device index (0, 1, 2, ...).
    y_offset:
        Vertical pixel offset to correct lens misalignment between the
        two sensors. Applied as a pure translation to the right frame.
        Typical value: 0 (factory-calibrated) or ±1–3 px.
    """

    # ZED outputs a side-by-side 1280×720 frame at 30 FPS
    _FRAME_W = 1280
    _FRAME_H = 720

    def __init__(self, camera_id: int = 0, y_offset: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_id}. "
                "Check that the ZED is connected and no other process is using it."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._FRAME_H)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimize internal buffer to reduce latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._y_offset = y_offset
        half_w = self._FRAME_W // 2
        # Pre-compute affine matrix for vertical correction (right frame only)
        self._shift_mat = np.float32([[1, 0, 0], [0, 1, float(y_offset)]])
        self._half_w = half_w

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture and return (left, right) BGR frames.

        Returns
        -------
        left, right:
            Each is shape ``(720, 640, 3)`` BGR.

        Raises
        ------
        RuntimeError
            If the camera fails to deliver a frame.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("ZED camera failed to deliver a frame.")

        left  = frame[:, :self._half_w, :]
        right = frame[:, self._half_w:, :]

        if self._y_offset != 0:
            right = cv2.warpAffine(
                right,
                self._shift_mat,
                (self._half_w, self._FRAME_H),
                flags=cv2.INTER_NEAREST,  # no resampling artefacts
            )

        return left, right

    def close(self) -> None:
        """Release the camera resource."""
        self._cap.release()

    def __enter__(self) -> "ZEDCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
