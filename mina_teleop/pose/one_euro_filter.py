"""One Euro Filter for real-time landmark smoothing"""

from __future__ import annotations
import math
import numpy as np

class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar value.

    Parameters
    ----------
    min_cutoff:
        Minimum cutoff frequency in Hz. Lower = smoother at rest, more lag.
    beta:
        Speed coefficient. Higher = more responsive during fast motion.
    d_cutoff:
        Cutoff frequency for the derivative filter (Hz).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev:  float | None = None
        self._dx_prev: float = 0.0
        self._t_prev:  float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Compute smoothing factor from cutoff frequency and timestep."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, x: float, t: float) -> float:
        """Filter a new sample.

        Parameters
        ----------
        x:
            Raw input value.
        t:
            Timestamp in seconds (monotonically increasing).

        Returns
        -------
        float
            Filtered value.
        """
        if self._t_prev is None:
            # First call — initialise state and return as-is
            self._x_prev = x
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev  # type: ignore[return-value]

        # Finite-difference derivative
        dx = (x - self._x_prev) / dt  # type: ignore[operator]

        # Filter the derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff — rises with motion speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the value
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev  # type: ignore[operator]

        # Store state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self) -> None:
        """Clear internal state (e.g. after tracking loss)."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class LandmarkFilter:
    """Applies OneEuroFilter independently to each axis of each landmark.

    Parameters
    ----------
    n_landmarks:
        Number of landmarks to filter (e.g. 33 for MediaPipe Pose).
    min_cutoff:
        Passed to each OneEuroFilter.
    beta:
        Passed to each OneEuroFilter.
    d_cutoff:
        Passed to each OneEuroFilter.
    """

    def __init__(
        self,
        n_landmarks: int,
        min_cutoff: float = 1.0,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
    ) -> None:
        self.n_landmarks = n_landmarks
        # One filter per landmark per axis (x, y, z)
        self._filters: list[list[OneEuroFilter]] = [
            [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]
            for _ in range(n_landmarks)
        ]

    def apply(self, landmarks: np.ndarray, t: float) -> np.ndarray:
        """Filter a landmark array.

        Parameters
        ----------
        landmarks:
            Shape ``(n_landmarks, 3)`` — raw positions.
        t:
            Current timestamp in seconds.

        Returns
        -------
        np.ndarray
            Shape ``(n_landmarks, 3)`` — filtered positions.
        """
        assert landmarks.shape == (self.n_landmarks, 3), (
            f"Expected shape ({self.n_landmarks}, 3), got {landmarks.shape}"
        )
        out = np.empty_like(landmarks)
        for i in range(self.n_landmarks):
            for j in range(3):
                out[i, j] = self._filters[i][j].apply(float(landmarks[i, j]), t)
        return out

    def reset(self) -> None:
        """Reset all filters (e.g. after tracking loss)."""
        for row in self._filters:
            for f in row:
                f.reset()
