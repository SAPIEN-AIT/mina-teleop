"""Abstract base class for all robot teleoperation controllers.

Every robot controller (LEAP hand, arm, future legs, …) implements this
interface. The teleoperation loop depends only on this interface, making
it trivial to swap robots or combine modalities.

Port of Binocular-Teleop/robots/base.py (Edgard, SAPIEN-AIT).

Example
-------
    from mina_teleop.robots.base import RobotController

    class LeapController(RobotController):
        def retarget(self, landmarks) -> np.ndarray:
            ...   # MediaPipe landmarks → 16 LEAP joint angles
            return q_16dof

        @classmethod
        def scene_xml_path(cls) -> str:
            from mina_assets import LEAP_HAND_SCENE
            return str(LEAP_HAND_SCENE)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np


class RobotController(ABC):
    """Abstract base for teleoperation controllers.

    A concrete subclass receives perception data every frame and returns a
    1-D array of target actuator values written to ``data.ctrl``.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def retarget(self, landmarks) -> np.ndarray:
        """Map perception input to robot actuator targets.

        Parameters
        ----------
        landmarks:
            Perception data — typically MediaPipe landmark list (21 entries).

        Returns
        -------
        np.ndarray
            1-D array of desired actuator values (length = model.nu).
        """
        ...

    @classmethod
    @abstractmethod
    def scene_xml_path(cls) -> str:
        """Absolute path to this robot's MuJoCo scene XML file."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (override as needed)
    # ------------------------------------------------------------------

    def init(
        self,
        data: mujoco.MjData,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
    ) -> None:
        """Called once before the simulation loop.

        Default: places mocap body ``hand_proxy`` at ``start_pos`` /
        ``start_quat`` and runs ``mj_forward``. Override for robots
        without a mocap body (e.g. fixed-base arm).
        """
        try:
            mid = self.model.body("hand_proxy").mocapid[0]
            data.mocap_pos[mid]  = start_pos
            data.mocap_quat[mid] = start_quat
            jid  = self.model.joint("palm_free").id
            addr = self.model.jnt_qposadr[jid]
            data.qpos[addr:addr + 3]     = start_pos
            data.qpos[addr + 3:addr + 7] = start_quat
        except (KeyError, IndexError):
            pass  # no mocap body — no-op
        mujoco.mj_forward(self.model, data)
