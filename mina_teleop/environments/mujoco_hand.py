"""MuJoCo LEAP hand simulation environment.

Architecture (from Binocular-Teleop, Edgard SAPIEN-AIT):
  - A ``hand_proxy`` mocap body is driven by vision (wrist 3-D position + orientation).
  - A soft weld equality constraint connects ``hand_proxy`` → ``palm``,
    so the hand follows the proxy with a 10 ms time constant — absorbs
    sudden jumps without stiff constraint explosions.
  - Finger joints are driven via position actuators (ctrl array).

Usage
-----
    from mina_teleop.environments.mujoco_hand import HandSim, HandSimConfig

    sim = HandSim()

    def callback(sim: HandSim) -> None:
        sim.set_wrist_pos(pos)
        sim.set_wrist_quat(quat)
        sim.set_finger_targets(angles)   # shape (16,)

    sim.run(teleop_callback=callback)

Run with mjpython (required on macOS for GLFW thread safety):
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
# Default scene
# ---------------------------------------------------------------------------

_DEFAULT_SCENE = LEAP_HAND_SCENE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FINGER_JOINTS = 16   # 4 fingers × 4 DOF

# Rest position of the hand proxy before physics starts
_DEFAULT_START_POS  = np.array([0.0, 0.30, 0.45])

# Base orientation: Rx(180°) → palm facing up (stable initial physics)
_DEFAULT_START_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# Relative pose of palm freejoint w.r.t. proxy at rest.
# Rx(-90°) applied to BASE_QUAT → fingers point +Z, palm faces -Y.
_RELPOSE_QUAT = np.array([0.5, -0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HandSimConfig:
    """Configuration for the LEAP hand simulation."""

    scene_path: Path = field(default_factory=lambda: _DEFAULT_SCENE)

    # Physics
    physics_hz:  float = 200.0
    n_substeps:  int   = 1      # physics steps per viewer frame (increase for speed)

    # Start pose
    start_pos:  np.ndarray = field(
        default_factory=lambda: _DEFAULT_START_POS.copy())
    start_quat: np.ndarray = field(
        default_factory=lambda: _DEFAULT_START_QUAT.copy())

    # Viewer camera
    cam_distance:  float = 0.6
    cam_azimuth:   float = 120.0
    cam_elevation: float = -20.0
    cam_lookat: tuple[float, float, float] = (0.0, 0.3, 0.3)


# ---------------------------------------------------------------------------
# HandSim
# ---------------------------------------------------------------------------

class HandSim:
    """Real-time LEAP hand simulation with mocap proxy control.

    The wrist position + orientation are controlled via a ``hand_proxy``
    mocap body. Fingers are controlled via position actuators.

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
                "Ensure mina_assets submodule is initialised:\n"
                "  git submodule update --init source/mina_assets"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_path))
        self.data  = mujoco.MjData(self.model)

        self.model.opt.timestep = 1.0 / self.cfg.physics_hz

        # Resolve mocap body id
        self._proxy_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy"
        )
        if self._proxy_id < 0:
            raise RuntimeError(
                "MJCF scene must contain a mocap body named 'hand_proxy'."
            )

        self._n_act = min(N_FINGER_JOINTS, self.model.nu)

        # Desired state (updated by set_* methods)
        self._finger_targets = np.zeros(self._n_act)
        self._wrist_pos  = self.cfg.start_pos.copy()
        self._wrist_quat = self.cfg.start_quat.copy()

        # Initialise physics at start pose
        self._init_hand()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_wrist_pos(self, pos: np.ndarray) -> None:
        """Set target wrist position (x, y, z) in world frame [m]."""
        self._wrist_pos[:] = pos

    def set_wrist_quat(self, quat: np.ndarray) -> None:
        """Set target wrist orientation as (w, x, y, z) quaternion."""
        self._wrist_quat[:] = quat

    def set_finger_targets(self, angles: np.ndarray) -> None:
        """Set desired finger joint angles.

        Parameters
        ----------
        angles:
            Shape ``(16,)`` — radians, ordered per IKRetargeter table.
        """
        self._finger_targets[:] = angles[: self._n_act]

    def reset(self) -> None:
        """Teleport hand back to start pose and clear finger targets."""
        self._finger_targets[:] = 0.0
        self._wrist_pos[:]  = self.cfg.start_pos
        self._wrist_quat[:] = self.cfg.start_quat
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self._init_hand()

    def run(
        self,
        teleop_callback: Callable[["HandSim"], None] | None = None,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Launch the MuJoCo viewer and run the simulation loop.

        Parameters
        ----------
        teleop_callback:
            Called once per viewer frame. Update wrist pose and finger
            targets inside this callback.
        key_callback:
            Optional GLFW key callback (keycode: int). Example: press A
            to trigger calibration in the teleop script.

        Must be called from the main thread (GLFW requirement on macOS).
        Use ``mjpython`` to launch.
        """
        cfg = self.cfg

        launch_kwargs: dict = {}
        if key_callback is not None:
            launch_kwargs["key_callback"] = key_callback

        with mujoco.viewer.launch_passive(
            self.model, self.data, **launch_kwargs
        ) as viewer:
            viewer.cam.distance  = cfg.cam_distance
            viewer.cam.azimuth   = cfg.cam_azimuth
            viewer.cam.elevation = cfg.cam_elevation
            viewer.cam.lookat[:] = cfg.cam_lookat

            print(
                "\n[HandSim] Running — close viewer to exit.\n"
                f"  Scene      : {cfg.scene_path}\n"
                f"  Actuators  : {self._n_act}\n"
                f"  Physics Hz : {cfg.physics_hz}\n"
                f"  Substeps   : {cfg.n_substeps}\n"
            )

            dt = self.model.opt.timestep * cfg.n_substeps

            while viewer.is_running():
                t_start = time.perf_counter()

                # 1. Teleoperation callback (user sets targets here)
                if teleop_callback is not None:
                    teleop_callback(self)

                # 2. Apply wrist pose to mocap proxy
                mid = self.data.model.body("hand_proxy").mocapid[0]
                self.data.mocap_pos[mid]  = self._wrist_pos
                self.data.mocap_quat[mid] = self._wrist_quat

                # 3. Apply finger targets (clamp to actuator limits)
                for i in range(self._n_act):
                    lo, hi = self.model.actuator_ctrlrange[i]
                    self.data.ctrl[i] = float(
                        np.clip(self._finger_targets[i], lo, hi)
                    )

                # 4. Physics steps
                for _ in range(cfg.n_substeps):
                    mujoco.mj_step(self.model, self.data)

                # 5. Render
                viewer.sync()

                # 6. Real-time pacing
                elapsed = time.perf_counter() - t_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_hand(self) -> None:
        """Teleport hand to start position before physics runs.

        Without this, the palm starts at its XML-defined origin and falls
        under gravity before the weld constraint can engage on frame 1.
        """
        mid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_proxy"
        )
        self.data.mocap_pos[mid]  = self.cfg.start_pos.copy()
        self.data.mocap_quat[mid] = self.cfg.start_quat.copy()

        # Also initialise the palm freejoint so physics starts at the right pose
        try:
            jid  = self.model.joint("palm_free").id
            addr = self.model.jnt_qposadr[jid]

            def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                aw, ax, ay, az = a
                bw, bx, by, bz = b
                return np.array([
                    aw*bw - ax*bx - ay*by - az*bz,
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                ])

            palm_q = _qmul(self.cfg.start_quat, _RELPOSE_QUAT)
            self.data.qpos[addr:addr + 3]     = self.cfg.start_pos
            self.data.qpos[addr + 3:addr + 7] = palm_q
        except (KeyError, IndexError):
            pass  # no freejoint — skip

        mujoco.mj_forward(self.model, self.data)
