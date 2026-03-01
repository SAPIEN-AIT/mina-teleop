"""Joint velocity limiter.

Prevents large instantaneous joint target jumps caused by tracking loss,
landmark discontinuities, or filter transients.
"""

from __future__ import annotations

import numpy as np


class VelocityLimiter:
    """Clamps how much each joint target can change per timestep.

    Parameters
    ----------
    max_rad_per_sec:
        Maximum allowed joint velocity in rad/s.
        2.0 rad/s is conservative and safe for most showcase scenarios.
    n_joints:
        Number of joints (default 10 for the bimanual arm).
    """

    def __init__(
        self,
        max_rad_per_sec: float = 2.0,
        n_joints: int = 10,
    ) -> None:
        self.max_rad_per_sec = max_rad_per_sec
        self._last: np.ndarray | None = None

    def apply(
        self,
        new_targets: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return clamped joint targets.

        Parameters
        ----------
        new_targets:
            Desired joint angles this step, shape ``(n_joints,)``.
        dt:
            Timestep in seconds since the last call.

        Returns
        -------
        np.ndarray
            Clamped targets, shape ``(n_joints,)``.
        """
        if self._last is None or dt <= 0.0:
            self._last = new_targets.copy()
            return new_targets.copy()

        max_delta = self.max_rad_per_sec * dt
        delta = new_targets - self._last
        clamped = self._last + np.clip(delta, -max_delta, max_delta)

        self._last = clamped.copy()
        return clamped

    def reset(self, targets: np.ndarray | None = None) -> None:
        """Reset internal state.

        Parameters
        ----------
        targets:
            If provided, initialise the last position to this value
            (useful after a teleop pause/resume to avoid a jump).
        """
        self._last = None if targets is None else targets.copy()
