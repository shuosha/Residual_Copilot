"""XArm environment for guided diffusion — accepts full 8D absolute actions.

The diffusion policy takes the pilot's base action as a reference signal during
denoising and outputs a complete 8D action (pos3 + quat4 + gripper1). This env
routes that action through the same admittance control + IK stack as XArmEnv,
skipping the residual combination step.
"""

import torch
import isaacsim.core.utils.torch as torch_utils  # type: ignore

from ..utils.control import adm_ctrl_task_space
from .xarm_env import XArmEnv


class XArmEnvGuidedDiffusion(XArmEnv):
    """XArmEnv variant that accepts 8D absolute Cartesian actions."""

    def _pre_physics_step(self, action):
        """Apply 8D absolute actions (pos + quat + gripper) with EMA smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.env_actions = action

        # Normalize quaternion after EMA blending.
        self.env_actions[:, 3:7] = self.env_actions[:, 3:7] / (
            self.env_actions[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        # Clamp gripper to [0, 1].
        self.env_actions[:, 7:8] = self.env_actions[:, 7:8].clamp(0.0, 1.0)

        # For reward compatibility (action_norm / action_smoothing penalties).
        self.residual_actions = self.env_actions[:, :self.cfg.action_space]

        # --- Admittance control + IK (identical to base class) ---
        ctrl_target_fingertip_midpoint_pos = self.env_actions[:, 0:3].clone()
        ctrl_target_fingertip_midpoint_quat = self.env_actions[:, 3:7].clone()

        ctrl_target_eef_pos = torch_utils.tf_combine(
            ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            -self.sim_fingertip2eef,
        )[1]

        cartesian_target, self.task_velocities = adm_ctrl_task_space(
            pos=self.eef_pos, quat=self.fingertip_midpoint_quat,
            pos_g=ctrl_target_eef_pos, quat_g=ctrl_target_fingertip_midpoint_quat,
            v=self.task_velocities, F_ext=self.F_ext, dt=self.physics_dt,
            kx=self.Kx, kr=self.Kr, mx=self.mx, mr=self.mr,
            dx=1. * torch.sqrt(self.Kx * self.mx), dr=1. * torch.sqrt(self.Kr * self.mr),
        )

        self.qpos_targets = self.ik_controller.compute_ik(
            init_qpos=self.joint_pos[:, 0:7],
            cartesian_target=cartesian_target,
        )
