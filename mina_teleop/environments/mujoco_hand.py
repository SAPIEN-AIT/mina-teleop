"""MuJoCo LEAP hand simulation environment.

Loads the LEAP hand MJCF scene and provides a real-time teleoperation
loop driven by a `teleop_callback`.

Architecture (from Binocular-Teleop, Edgard SAPIEN-AIT):
  - A `hand_proxy` mocap body is driven by vision (wrist 3D position).
  - A soft weld equality constraint connects `hand_proxy` → `palm`,
    so the hand follows the proxy with a 10 ms time constant — this
    absorbs sudden jumps without stiff constraint explosions.
  - Finger joints are driven via position actuators (ctrl array).

Usage
-----
    from mina_teleop.environments.mujoco_hand import HandSim, HandSimConfig
    from mina_teleop.retargeters.hand import LeapRetargeter

    sim = HandSim()
    retargeter = LeapRetargeter()

    def callback(sim: HandSim) -> None:
        lm = my_detector.get_landmarks()
        if lm is not None:
            angles = retargeter.retarget(lm)
            sim.set_finger_targets(angles)
            sim.set_wrist_pos(my_wrist_pos)

    sim.run(teleop_callback=callback)

Run with mjpython (required on macOS):
    mjpython scripts/teleop/teleop_sim_hand.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np
from mina_assets import LEAP_HAND_SCENE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_SCENE = LEAP_HAND_SCENE

# ---------------------------------------------------------------------------
# Number of LEAP finger actuators
# ---------------------------------------------------------------------------

N_FINGER_JOINTS = 16  # 4 fingers × 4 DOF


@dataclass
class HandSimConfig:
    """Configuration for the LEAP hand simulation."""

    scene_path: Path = field(default_factory=lambda: _DEFAULT_SCENE)

    # Physics
    physics_hz: float = 200.0   # simulation rate

    # Viewer camera
    cam_distance:  float = 0.6
    cam_azimuth:   float = 120.0
    cam_elevation: float = -20.0
    cam_lookat: tuple[float, float, float] = (0.0, 0.3, 0.3)


class HandSim:
    """Real-time LEAP hand simulation with mocap proxy control.

    The wrist position is controlled via a `hand_proxy` mocap body.
    Fingers are controlled via position actuators.

    Parameters
    ----------
    cfg:
        Simulation configuration. Pass ``HandSimConfig()`` for defaults.
    """

    def __init__(self, cfg: HandSimConfig | None = None) -> None:
        self.cfg = cfg or HandSimConfig()

        if not self.cfg.scene_path.exists():
            raise FileNotFoundError(
                f"LEAP scene not found: {self.cfg.scene_path}\n"
                "Copy mjcf/leap_hand/ from Binocular-Teleop into the Mina repo."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_path))
        self.data  = mujoco.MjData(self.model)

        self.model.opt.timestep = 1.0 / self.cfg.physics_hz

        # Resolve mocap body id for hand_proxy
        self._proxy_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy"
        )
        if self._proxy_id < 0:
            raise RuntimeError(
                "MJCF scene must contain a mocap body named 'hand_proxy'."
            )

        # Finger actuator ids (first N_FINGER_JOINTS actuators in the model)
        # NOTE: verify ordering matches retargeters/hand.py joint table.
        self._n_act = min(N_FINGER_JOINTS, self.model.nu)

        # Desired finger angles
        self._finger_targets = np.zeros(self._n_act)

        # Desired wrist pose
        self._wrist_pos  = self.data.mocap_pos[0].copy()
        self._wrist_quat = self.data.mocap_quat[0].copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_wrist_pos(self, pos: np.ndarray) -> None:
        """Set target wrist position (x, y, z) in world frame."""
        self._wrist_pos[:] = pos

    def set_wrist_quat(self, quat: np.ndarray) -> None:
        """Set target wrist orientation as wxyz quaternion."""
        self._wrist_quat[:] = quat

    def set_finger_targets(self, angles: np.ndarray) -> None:
        """Set desired finger joint angles.

        Parameters
        ----------
        angles:
            Shape ``(16,)`` — radians, ordered per LeapRetargeter table.
        """
        self._finger_targets[:] = angles[: self._n_act]

    def run(
        self,
        teleop_callback: Callable[["HandSim"], None] | None = None,
    ) -> None:
        """Launch the MuJoCo viewer and run the simulation loop.

        Parameters
        ----------
        teleop_callback:
            Called once per physics step. Update wrist pose and finger
            targets inside this callback::

                def my_cb(sim: HandSim) -> None:
                    sim.set_wrist_pos(pos)
                    sim.set_finger_targets(angles)

        Must be called from the main thread (GLFW requirement on macOS).
        Use ``mjpython`` to launch.
        """
        cfg = self.cfg

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance  = cfg.cam_distance
            viewer.cam.azimuth   = cfg.cam_azimuth
            viewer.cam.elevation = cfg.cam_elevation
            viewer.cam.lookat[:] = cfg.cam_lookat

            print(
                "\n[HandSim] Running — close viewer to exit.\n"
                f"  Scene : {cfg.scene_path}\n"
                f"  DOF   : {self._n_act} finger actuators\n"
                f"  Hz    : {cfg.physics_hz}\n"
            )

            dt = self.model.opt.timestep

            while viewer.is_running():
                t_start = time.perf_counter()

                # 1. Teleoperation callback
                if teleop_callback is not None:
                    teleop_callback(self)

                # 2. Apply wrist pose to mocap proxy
                self.data.mocap_pos[0]  = self._wrist_pos
                self.data.mocap_quat[0] = self._wrist_quat

                # 3. Apply finger targets to actuators
                self.data.ctrl[: self._n_act] = self._finger_targets

                # 4. Physics step
                mujoco.mj_step(self.model, self.data)

                # 5. Render
                viewer.sync()

                # 6. Real-time pacing
                elapsed = time.perf_counter() - t_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
