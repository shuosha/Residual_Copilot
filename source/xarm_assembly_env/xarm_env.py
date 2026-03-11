# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

import numpy as np
import torch

import carb # type: ignore
import isaacsim.core.utils.torch as torch_utils # type: ignore

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.markers import VisualizationMarkers

from ..utils.control import adm_ctrl_task_space, IK_Controller
from ..utils.utils import (
    resolve_hf, build_init_state, collapse_obs_dict, set_friction,
    quat_geodesic_angle, get_held_base_pose, get_target_held_base_pose,
)
from .xarm_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, XArmEnvCfg
from .assembly_tasks_cfg import HF_ASSETS_REPO, HF_MODELS_REPO, HF_DATA_REPO

from residual_copilot.pilot_models.knn_pilot import KNN_Pilot
from residual_copilot.pilot_models.bc_pilot import BC_Pilot

_KNN_CFG_PATH = str(Path(__file__).parent.parent / "pilot_models/config/knn_cfg.json")


class XArmEnv(DirectRLEnv):
    cfg: XArmEnvCfg

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, cfg: XArmEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._init_pilot()

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, -0.0015), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset) # type: ignore
        self._held_asset = Articulation(self.cfg_task.held_asset) # type: ignore
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg) # type: ignore
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg) # type: ignore

        self.eef_contact_sensor = ContactSensor(self.cfg.eef_contact_sensor_cfg)
        self.scene.sensors["eef_contact_sensor"] = self.eef_contact_sensor

        self.held_asset_contact_sensor = ContactSensor(self.cfg_task.held_asset_contact_sensor_cfg)
        self.scene.sensors["held_asset_contact_sensor"] = self.held_asset_contact_sensor

        if self.cfg.vis.store_rgb:
            self.front_camera = TiledCamera(self.cfg.front_camera_cfg)
            self.scene.sensors["front_camera"] = self.front_camera

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.vis.vis_obs:
            self.held_asset_marker = VisualizationMarkers(self.cfg.vis.frame_marker_cfg)
            self.fixed_asset_marker = VisualizationMarkers(self.cfg.vis.frame_marker_cfg)

    def _init_tensors(self):
        """Initialize control and state tensors."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset.
        self.held_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_held_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Trajectory geometry augmentation.
        self.xy_translation_noise = torch.zeros((self.num_envs, 2), device=self.device)
        self.yaw_rotation_noise = torch.zeros((self.num_envs, 1), device=self.device)

        self.held_center_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "gear_mesh":
            self.held_center_pos_local[:, 0] += self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]
            self.held_center_pos_local[:, 2] += self.cfg_task.held_asset_cfg.grasp_offset[2]
        elif self.cfg_task.name == "peg_insert":
            self.held_center_pos_local[:, 2] += self.cfg_task.held_asset_cfg.height
            self.held_center_pos_local[:, 2] -= self.cfg_task.held_asset_cfg.grasp_offset[2]

        # Body indices.
        self.eef_body_idx = self._robot.body_names.index("link7")
        self.arm_dof_idx, _ = self._robot.find_joints("joint.*")
        self.gripper_dof_idx, _ = self._robot.find_joints("gripper")

        # sim2real offsets.
        self.sim_fingertip2eef = torch.tensor([self.cfg_task.robot_cfg.sim_fingertip2eef], device=self.device).repeat(self.num_envs, 1)
        self.real_fingertip2eef = torch.tensor([self.cfg_task.robot_cfg.real_fingertip2eef], device=self.device).repeat(self.num_envs, 1)

        self.eef_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self.task_velocities = torch.zeros((self.num_envs, 6), device=self.device)

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.first_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.rolling_success_rate = 0.0
        self.ema_alpha = 0.002  # ~350 ep half-life for 450-ts episodes

        self.residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.residual_actions)
        self.env_actions = torch.zeros((self.num_envs, 8), device=self.device)

        self.rew_sum = None

        self.noise_gate = torch.zeros(self.num_envs, 1, device=self.device)
        self.starting_qpos = None
        self.curr_decimation = 0

        # Admittance control parameters (per-env, randomized in _reset_idx when dmr is on).
        self.Kx = torch.tensor([self.cfg.dmr.Kx], device=self.device).repeat(self.num_envs)
        self.Kr = torch.tensor([self.cfg.dmr.Kr], device=self.device).repeat(self.num_envs)
        self.mx = torch.tensor([self.cfg.dmr.mx], device=self.device).repeat(self.num_envs)
        self.mr = torch.tensor([self.cfg.dmr.mr], device=self.device).repeat(self.num_envs)

        # IK controller.
        robot_dir = resolve_hf(HF_ASSETS_REPO, "robot", materialize=True)
        self.ik_controller = IK_Controller(urdf_path=str(Path(robot_dir) / "xarm7.urdf"))

        # Nut-thread-specific yaw tracking.
        if self.cfg_task.name == "nut_thread":
            self.prev_held_yaw = torch.zeros(self.num_envs, device=self.device)
            self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
            self.picked_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_pilot(self):
        """Load pilot model and initial episode states from HuggingFace."""
        if self.cfg.pilot_model in ["bc_teleop", "bc_expert"]:
            ckpt_path = self.cfg_task.dp_expert_path if self.cfg.pilot_model == "bc_expert" else self.cfg_task.dp_teleop_path
            self.pilot = BC_Pilot(resolve_hf(HF_MODELS_REPO, ckpt_path, repo_type="model"))
        elif self.cfg.pilot_model in ["knn", "replay"]:
            self.pilot = KNN_Pilot(
                cfg_path=_KNN_CFG_PATH,
                data_path=resolve_hf(HF_DATA_REPO, self.cfg_task.train_data_path),
                num_envs=self.num_envs,
                device=self.device,
                replay_mode=(self.cfg.pilot_model == "replay"),
            )
        else:
            raise ValueError(f"Unknown pilot model: {self.cfg.pilot_model}")

        self.add_lag_to_base = self.cfg.pilot_type == "laggy"
        self.add_noise_to_base = self.cfg.pilot_type == "noisy"

        self.base_actions = torch.zeros((self.num_envs, 8), device=self.device)
        self.last_base_actions = torch.zeros((self.num_envs, 8), device=self.device)
        self.has_last_base = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Initial states (fingertip pose + fixed/held positions per episode).
        self.initial_poses = build_init_state(
            data_path=resolve_hf(HF_DATA_REPO, self.cfg_task.train_data_path),
            num_envs=self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )  # (num_envs, num_eps, 13)
        self.total_episodes: int = self.initial_poses.shape[1]

        if self.cfg.vis.order_envs:
            self.episode_idx = torch.arange(0, self.num_envs, device=self.device) % self.total_episodes
        else:
            self.episode_idx = torch.randint(0, self.total_episodes, (self.num_envs,), device=self.device)

    def _set_default_dynamics_parameters(self):
        """Set friction and action threshold tensors."""
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.gripper_threshold = torch.tensor(self.cfg.ctrl.gripper_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    # -------------------------------------------------------------------------
    # Step loop — Observations
    # -------------------------------------------------------------------------

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        self.last_base_actions[self.has_last_base] = self.base_actions.clone()[self.has_last_base]

        if self.cfg.pilot_model in ["knn", "replay"]:
            self._get_knn_pilot_action()
        elif self.cfg.pilot_model in ["bc_teleop", "bc_expert"]:
            self._get_bc_pilot_action()

        self.last_base_actions[~self.has_last_base] = self.base_actions.clone()[~self.has_last_base]
        self.has_last_base.fill_(True)

        if self.add_noise_to_base:
            self._add_noise_to_base()
        if self.add_lag_to_base:
            self._add_lag_to_base()

        obs_dict, state_dict = self._get_obs_state_dict()

        obs_tensors = collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        self._visualize_markers()

        return {"policy": obs_tensors, "critic": state_tensors}

    def _get_knn_pilot_action(self):
        sim_fingertip_pos = self.fingertip_midpoint_pos.clone()
        sim_fingertip_pos[:, :2] -= self.xy_translation_noise
        sim_eef_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat.clone(),
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((self.num_envs,), device=self.device),
                pitch=torch.zeros((self.num_envs,), device=self.device),
                yaw=-self.yaw_rotation_noise.squeeze(-1),
            ),
        )

        self.base_actions = self.pilot.get_actions(
            self.episode_idx, sim_fingertip_pos, sim_eef_quat, self.gripper, verbose=False
        )

        self.base_actions[:, :2] += self.xy_translation_noise
        self.base_actions[:, 3:7] = torch_utils.quat_mul(
            self.base_actions[:, 3:7],
            torch_utils.quat_from_euler_xyz(
                roll=torch.zeros((self.num_envs,), device=self.device),
                pitch=torch.zeros((self.num_envs,), device=self.device),
                yaw=self.yaw_rotation_noise.squeeze(-1),
            ),
        )

    def _get_bc_pilot_action(self):
        bc_obs = {
            "observation.state": torch.cat([
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.gripper,
                self.ee_linvel_fd,
                self.ee_angvel_fd,
            ], dim=-1),
            "observation.environment_state": torch.cat([
                self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
                self.fingertip_midpoint_pos - self.held_pos_obs_frame,
            ], dim=-1),
        }
        self.base_actions = self.pilot.act(bc_obs)

    def _add_noise_to_base(self):
        N, d = self.num_envs, self.cfg.action_space
        lo, hi = self.cfg.dmr.base_noise_range
        p_on = self.cfg.dmr.base_noise_prob
        beta = self.cfg.dmr.base_noise_gate_alpha

        eps = torch.empty(N, d, device=self.device).uniform_(lo, hi)
        eps[:, -1] = 0.0  # never noise the gripper

        g_tgt = (torch.rand(N, 1, device=self.device) < p_on).float()
        self.noise_gate = beta * self.noise_gate + (1.0 - beta) * g_tgt

        self.base_actions = self._apply_residual(self.noise_gate * eps, self.base_actions)

    def _add_lag_to_base(self):
        lag_gate = (torch.rand(self.num_envs, 1, device=self.device) < self.cfg.dmr.base_lag_prob).float()
        self.base_actions = lag_gate * self.last_base_actions + (1.0 - lag_gate) * self.base_actions

    def _get_obs_state_dict(self):
        """Build observation and state dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        noisy_held_pos = self.held_pos_obs_frame + self.init_held_pos_obs_noise
        prev_actions = self.residual_actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_pos_rel_held": self.fingertip_midpoint_pos - noisy_held_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "gripper": self.gripper,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
            "base_fingertip_pos": self.base_actions[:, :3],
            "base_fingertip_quat": self.base_actions[:, 3:7],
            "base_gripper": self.base_actions[:, 7:8],
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_pos_rel_held": self.fingertip_midpoint_pos - self.held_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "gripper": self.gripper,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
            "base_fingertip_pos": self.base_actions[:, :3],
            "base_fingertip_quat": self.base_actions[:, 3:7],
            "base_gripper": self.base_actions[:, 7:8],
        }

        return obs_dict, state_dict

    def _compute_intermediate_values(self, dt):
        """Compute derived quantities from raw simulator state."""
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.held_pos_obs_frame = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            self.held_center_pos_local,
        )[1]

        self.eef_pos = self._robot.data.body_pos_w[:, self.eef_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.eef_body_idx]
        self.fingertip_midpoint_pos = torch_utils.tf_combine(
            self.fingertip_midpoint_quat,
            self.eef_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
            self.sim_fingertip2eef,
        )[1]

        self.gripper = self._robot.data.joint_pos[:, self.gripper_dof_idx[0:1]] / 1.6
        self.gripper = torch.clamp(self.gripper, 0.0, self.cfg_task.gripper_obs_clamp)

        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.eef_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.eef_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.eef_jacobian = jacobians[:, self.eef_body_idx - 1, 0:6, 0:7]

        self.eef_force = self.eef_contact_sensor.data.net_forces_w.squeeze(1)
        self.F_ext = torch.cat([self.eef_force, torch.zeros((self.num_envs, 3), device=self.device)], dim=-1)
        self.held_asset_force = self.held_asset_contact_sensor.data.net_forces_w.squeeze(1)

        if self.cfg.vis.store_rgb:
            self.front_rgb = self.front_camera.data.output["rgb"]

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differenced velocities (more reliable than physics body velocities).
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        self.ee_angvel_fd = axis_angle_from_quat(rot_diff_quat) / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

        if self.cfg_task.name == "nut_thread":
            self.picked_up = self.picked_up | (self.held_pos[:, 2] > 0.03)

    # -------------------------------------------------------------------------
    # Step loop — Actions
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, action):
        """Apply policy actions with EMA smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.residual_actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.residual_actions

        self.env_actions = self._apply_residual(self.residual_actions, self.base_actions)

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

    def _apply_residual(self, residual_actions, base_actions):
        """Combine base and residual actions into a Cartesian target."""
        pos_actions = residual_actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = base_actions[:, 0:3] + pos_actions

        rot_actions = residual_actions[:, 3:6] * self.rot_threshold
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, base_actions[:, 3:7])

        grip_actions = residual_actions[:, 6:7] * self.gripper_threshold
        ctrl_target_gripper_dof_pos = torch.clamp(base_actions[:, 7:8] + grip_actions, 0.0, 1.0)

        return torch.cat([
            ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos,
        ], dim=-1)

    def _apply_action(self):
        """Apply joint targets to the robot, interpolating within decimation."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        ctrl_target_gripper_dof_pos = self.env_actions[:, 7:8].clone() * 1.6
        ctrl_target_gripper_dof_pos = torch.clamp(ctrl_target_gripper_dof_pos, max=self.cfg_task.gripper_ctrl_clamp)

        if self.starting_qpos is None:
            self.starting_qpos = self.joint_pos[:, :7].clone()
        ratio = (self.curr_decimation + 1) / self.cfg.decimation
        qpos_target = ratio * self.qpos_targets + (1.0 - ratio) * self.starting_qpos
        self.curr_decimation += 1

        if self.curr_decimation == self.cfg.decimation:
            self.starting_qpos = None
            self.curr_decimation = 0

        self._robot.set_joint_position_target(qpos_target, joint_ids=self.arm_dof_idx)
        self._robot.set_joint_position_target(ctrl_target_gripper_dof_pos, joint_ids=self.gripper_dof_idx)

    # -------------------------------------------------------------------------
    # Step loop — Done / Reward
    # -------------------------------------------------------------------------

    def _get_dones(self):
        """Check termination conditions and update rolling success rate."""
        self._compute_intermediate_values(dt=self.physics_dt)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        task_success, assembly_error = self._check_success()
        self.assembly_error = assembly_error
        self.first_success = task_success & (~self.ep_succeeded.to(torch.bool))
        self.ep_succeeded[task_success] = 1

        terminated = torch.norm(self.fingertip_midpoint_pos - self.held_pos_obs_frame, dim=1) > 0.2
        terminated |= self.ep_succeeded.bool()

        if self.cfg_task.name == "peg_insert":
            unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            tilt_degrees = quat_geodesic_angle(self.held_quat, unit_quat) * 180.0 / math.pi
            terminated |= tilt_degrees > 60.0

        if self.cfg_task.name == "nut_thread":
            terminated |= (self.held_pos[:, 2] < 0.02) & self.picked_up

        if self.cfg.pilot_model == "replay":
            terminated |= self.pilot.replay_done(self.episode_idx)

        assert not (self.first_success & (~terminated)).any()

        done = torch.logical_or(time_out, terminated)
        s = self.ep_succeeded[done].float()
        n = s.numel()
        if n > 0:
            alpha = self.ema_alpha
            exponents = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.float32)
            weights = (1.0 - alpha) ** exponents
            self.rolling_success_rate = (
                (1.0 - alpha) ** n * self.rolling_success_rate + alpha * (weights * s).sum()
            )

        self.extras["rolling_avg_succ_rate"] = float(self.rolling_success_rate)
        self.extras["assembly_error"] = float(assembly_error.mean())

        return terminated, time_out

    def _check_success(self):
        """Check task success; nut_thread uses cumulative rotation instead of position.

        Returns:
            assembled: bool tensor (num_envs,) indicating per-env success.
            assembly_error: float tensor (num_envs,) continuous assembly error.
                For peg_insert/gear_mesh: ||held_pos - target_pos||.
                For nut_thread: |success_rotation_threshold_deg - cumulative_rotation|.
        """
        assembled = self._get_assembly_status(self.cfg_task.success_threshold)
        held_pos, target_pos = self._get_held_target_pos()

        if self.cfg_task.name == "nut_thread":
            xy_dist = torch.linalg.vector_norm(target_pos[:, :2] - held_pos[:, :2], dim=1)
            z_disp = held_pos[:, 2] - target_pos[:, 2]
            thread_precondition = (xy_dist < 0.015) & (z_disp < 0.01)
            gripper_closed = self.gripper.squeeze(-1) >= self.cfg_task.close_gripper

            _, _, yaw = torch_utils.get_euler_xyz(self.held_quat)
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            dyaw = (yaw - self.prev_held_yaw + math.pi) % (2 * math.pi) - math.pi
            dyaw_deg = dyaw * (180.0 / math.pi)

            acc_mask = thread_precondition & gripper_closed
            self.cumulative_rotation[acc_mask] += torch.clamp(-dyaw_deg, min=0.0)[acc_mask]
            self.prev_held_yaw = yaw.clone()

            assembled = self.cumulative_rotation >= self.cfg_task.success_rotation_threshold_deg
            assembly_error = torch.abs(self.cfg_task.success_rotation_threshold_deg - self.cumulative_rotation)
        else:
            assembly_error = torch.linalg.vector_norm(held_pos - target_pos, dim=1)

        return assembled, assembly_error

    def _get_held_target_pos(self):
        """Return (held_pos, target_pos) base positions for the current task."""
        held_pos, _ = get_held_base_pose(
            self.held_pos, self.held_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device,
        )
        target_pos, _ = get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device,
        )
        return held_pos, target_pos

    def _get_assembly_status(self, success_threshold):
        """Check per-env success based on XY centering and height threshold."""
        held_pos, target_pos = self._get_held_target_pos()

        xy_dist = torch.linalg.vector_norm(target_pos[:, :2] - held_pos[:, :2], dim=1)
        z_disp = held_pos[:, 2] - target_pos[:, 2]
        is_centered = xy_dist < 0.0025

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = success_threshold
        else:
            raise NotImplementedError(f"Task '{self.cfg_task.name}' not implemented")

        return is_centered & (z_disp < height_threshold)

    def _get_rewards(self):
        """Compute dense shaping + terminal rewards."""

        # XY align reward.
        held_pos, target_pos = self._get_held_target_pos()
        xy_align_thresh = 0.005
        xy_aligned = (
            (torch.linalg.vector_norm(target_pos[:, :2] - held_pos[:, :2], dim=1) < xy_align_thresh).float()
            + (torch.linalg.vector_norm(self.fingertip_midpoint_pos[:, :2] - held_pos[:, :2], dim=1) < xy_align_thresh).float()
        )

        # Tilt penalty.
        a_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        a_world = torch_utils.quat_rotate(self.env_actions[:, 3:7], a_local)
        z_down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(a_world)
        tilt_penalty = torch.acos((a_world * z_down).sum(dim=-1).clamp(-1.0, 1.0))

        # Contact-force penalty.
        F = torch.norm(self.held_asset_force, dim=1)
        force_penalty = torch.clamp((F - 10.0).clamp_min(0.0) / 20.0, max=1.0)

        # Action penalties.
        action_norm = torch.norm(self.residual_actions, dim=1) / math.sqrt(self.cfg.action_space)
        action_smoothing = torch.norm(self.prev_actions - self.residual_actions, dim=1)

        rew_dict = {
            "action_norm":      -action_norm      * self.cfg_task.action_norm_reward_scale,
            "tilt_penalty":     -tilt_penalty      * self.cfg_task.tilt_penalty_reward_scale,
            "force_penalty":    -force_penalty     * self.cfg_task.force_penalty_reward_scale,
            "action_smoothing": -action_smoothing  * self.cfg_task.action_smoothing_reward_scale,
            "xy_align":          xy_aligned.float() * self.cfg_task.xy_aligned_reward_scale,
            "terminated":       -(self.reset_buf & (~self.ep_succeeded)).float() * self.cfg_task.termination_reward_scale,
            "task_success":      self.first_success.float() * self.cfg_task.task_success_reward_scale,
        }

        rew_buf = sum(rew_dict.values())

        self.prev_actions = self.residual_actions.clone()
        self._log_metrics(rew_dict)

        return rew_buf

    def _log_metrics(self, rew_dict):
        """Log per-step reward components and first-success timing."""
        first_success_ids = self.first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)
        if len(nonzero_success_ids) > 0:
            self.extras["success_times"] = (
                self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        if self.rew_sum is None:
            self.rew_sum = {k: 0.0 for k in rew_dict}

        LOG_INTERVAL = 100
        for rew_name, rew in rew_dict.items():
            self.rew_sum[rew_name] += rew.mean().item()

        if self.cfg.vis.print_rew and self.common_step_counter % LOG_INTERVAL == 0:
            GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
            print("\n" + "=" * 50)
            print(f" Iter {self.common_step_counter // LOG_INTERVAL}")
            print("=" * 50)
            mean_rew = 0.0
            for rew_name, rew in self.rew_sum.items():
                val = rew / LOG_INTERVAL
                color = GREEN if val >= 0 else RED
                print(f"{rew_name}: {color}{val:.4f}{RESET}")
                self.rew_sum[rew_name] = 0.0
                mean_rew += val
            print("mean rew: ", mean_rew)

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        """Reset given environments."""
        super()._reset_idx(env_ids)

        if self.cfg.vis.store_rgb:
            self.front_camera.reset(env_ids=env_ids)

        self._reset_dmr_params(env_ids)
        translation_noise, _, fixed_height_noise, yaw_delta_quat, identity_quat = (
            self._reset_observation_noise(env_ids)
        )
        fixed_tip_pos_local = self._reset_asset_poses(
            env_ids, translation_noise, yaw_delta_quat, fixed_height_noise, identity_quat,
        )
        self._reset_robot_pose(env_ids, translation_noise, yaw_delta_quat, identity_quat)
        self._reset_pilot_and_buffers(env_ids, identity_quat, fixed_tip_pos_local)

    def _reset_dmr_params(self, env_ids):
        """Randomize admittance control parameters and advance episode index."""
        if self.cfg.dmr.rand_ctrl:
            lo, hi = self.cfg.dmr.Kx_range
            self.Kx[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
            lo, hi = self.cfg.dmr.Kr_range
            self.Kr[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
            lo, hi = self.cfg.dmr.mx_range
            self.mx[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
            lo, hi = self.cfg.dmr.mr_range
            self.mr[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        self.task_velocities[env_ids] = 0.0
        self.episode_idx[env_ids] = (self.episode_idx[env_ids] + 1) % self.total_episodes

    def _reset_observation_noise(self, env_ids):
        """Generate observation noise and data augmentation noise.

        Returns:
            (translation_noise, yaw_rotation_noise, fixed_height_noise,
             yaw_delta_quat, identity_quat)
        """
        n = len(env_ids)

        # Observation noise.
        fixed_noise = torch.randn((n, 3), device=self.device)
        fixed_noise = fixed_noise @ torch.diag(torch.tensor(self.cfg.dmr.fixed_asset_pos, device=self.device))
        self.init_fixed_pos_obs_noise[env_ids] = fixed_noise

        held_noise = torch.randn((n, 3), device=self.device)
        held_noise = held_noise @ torch.diag(torch.tensor(self.cfg.dmr.held_asset_pos, device=self.device))
        self.init_held_pos_obs_noise[env_ids] = held_noise

        # Data augmentation.
        if self.cfg.dmr.aug_data:
            translation_noise = torch.randn((n, 2), device=self.device) * self.cfg.dmr.pos_xy_aug
            yaw_rotation_noise = torch.randn((n,), device=self.device) * math.radians(self.cfg.dmr.rot_aug)
            fixed_height_noise = torch.randn((n,), device=self.device) * self.cfg.dmr.pos_z_aug
        else:
            translation_noise = torch.zeros((n, 2), device=self.device)
            yaw_rotation_noise = torch.zeros((n,), device=self.device)
            fixed_height_noise = torch.zeros((n,), device=self.device)

        self.xy_translation_noise[env_ids] = translation_noise
        self.yaw_rotation_noise[env_ids] = yaw_rotation_noise.unsqueeze(-1)

        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(n, -1)
        yaw_delta_quat = torch_utils.quat_from_euler_xyz(
            roll=torch.zeros(n, device=self.device),
            pitch=torch.zeros(n, device=self.device),
            yaw=yaw_rotation_noise,
        )

        return translation_noise, yaw_rotation_noise, fixed_height_noise, yaw_delta_quat, identity_quat

    def _reset_asset_poses(self, env_ids, translation_noise, yaw_delta_quat, fixed_height_noise, identity_quat):
        """Compute and set held/fixed asset poses. Returns fixed_tip_pos_local."""
        n = len(env_ids)

        # Held asset pose.
        held_pos = self.initial_poses[env_ids, self.episode_idx[env_ids], -3:]
        held_pos = torch_utils.tf_combine(
            identity_quat, held_pos, identity_quat, -self.held_center_pos_local[env_ids],
        )[1]
        held_pos[:, :2] += translation_noise
        held_quat = torch_utils.quat_mul(identity_quat, yaw_delta_quat)

        # Fixed asset pose.
        fixed_tip_pos_local = torch.zeros((n, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]  # type: ignore

        fixed_pos = self.initial_poses[env_ids, self.episode_idx[env_ids], -6:-3]
        fixed_pos = torch_utils.tf_combine(
            identity_quat, fixed_pos, identity_quat, -fixed_tip_pos_local,
        )[1]
        fixed_pos[:, :2] += translation_noise
        fixed_pos[:, 2] += fixed_height_noise
        fixed_quat = torch_utils.quat_mul(identity_quat, yaw_delta_quat)

        self._set_assets_state(
            held_pos=held_pos, held_quat=held_quat,
            fixed_pos=fixed_pos, fixed_quat=fixed_quat,
            env_ids=env_ids,
        )

        return fixed_tip_pos_local

    def _reset_robot_pose(self, env_ids, translation_noise, yaw_delta_quat, identity_quat):
        """Solve IK for initial robot pose and apply it."""
        init_qpos = self._robot.data.default_joint_pos[env_ids, :7]
        init_fingertip = self.initial_poses[env_ids, self.episode_idx[env_ids], :7]
        sim_eef = init_fingertip.clone()
        sim_eef[:, :2] += translation_noise
        sim_eef[:, 3:7] = torch_utils.quat_mul(sim_eef[:, 3:7], yaw_delta_quat)
        sim_eef[:, 0:3] = torch_utils.tf_combine(
            sim_eef[:, 3:7], sim_eef[:, 0:3], identity_quat, -self.sim_fingertip2eef[env_ids],
        )[1]
        noised_qpos = self.ik_controller.compute_ik(init_qpos=init_qpos, cartesian_target=sim_eef[:, :7])
        self._set_default_robot_pose(joints=noised_qpos, env_ids=env_ids)

    def _reset_pilot_and_buffers(self, env_ids, identity_quat, fixed_tip_pos_local):
        """Reset pilot model, recompute fixed_pos_obs_frame, and clear action buffers."""
        # Reset pilot.
        if self.cfg.pilot_model in ["bc_teleop", "bc_expert"]:
            self.pilot.reset()
        elif self.cfg.pilot_model in ["knn", "replay"]:
            self.pilot.clear(env_ids)
        self.has_last_base[env_ids] = False

        if self.add_noise_to_base:
            self.noise_gate[env_ids] = 0.0

        # Recompute fixed_pos_obs_frame after placing assets.
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat[env_ids], self.fixed_pos[env_ids], identity_quat, fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[env_ids] = fixed_tip_pos

        # Reset velocity / action buffers.
        self.prev_fingertip_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.prev_fingertip_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()
        self.ee_angvel_fd[env_ids, :] = 0.0
        self.ee_linvel_fd[env_ids, :] = 0.0
        self.residual_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.env_actions[env_ids] = 0.0

        # Reset nut-thread yaw tracking.
        if self.cfg_task.name == "nut_thread":
            _, _, held_yaw0 = torch_utils.get_euler_xyz(self.held_quat[env_ids])
            self.prev_held_yaw[env_ids] = (held_yaw0 + math.pi) % (2 * math.pi) - math.pi
            self.cumulative_rotation[env_ids] = 0.0
            self.picked_up[env_ids] = False

    def _reset_buffers(self, env_ids):
        """Reset per-episode tracking buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0
        self.first_success[env_ids] = False

    def _set_assets_state(self, held_pos, held_quat, fixed_pos, fixed_quat, env_ids):
        """Teleport assets to their reset poses."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        fixed_state = torch.zeros((len(env_ids), 13), device=self.device)
        fixed_state[:, 0:3] = fixed_pos + self.scene.env_origins[env_ids]
        fixed_state[:, 3:7] = fixed_quat
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        self.step_sim_no_action()

        if self.cfg_task.name == "gear_mesh":
            for gear_asset in (self._small_gear_asset, self._large_gear_asset):
                gear_state = gear_asset.data.default_root_state.clone()[env_ids]
                gear_state[:, 0:7] = fixed_state[:, 0:7]
                gear_state[:, 7:] = 0.0
                gear_asset.write_root_pose_to_sim(gear_state[:, 0:7], env_ids=env_ids)
                gear_asset.write_root_velocity_to_sim(gear_state[:, 7:], env_ids=env_ids)
                gear_asset.reset(env_ids=env_ids)

        held_state = torch.zeros((len(env_ids), 13), device=self.device)
        held_state[:, 0:3] = held_pos + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = held_quat
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset(env_ids=env_ids)

        self.step_sim_no_action()

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _set_default_robot_pose(self, joints, env_ids):
        """Set robot to a given joint configuration."""
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7] = 0.0   # gripper
        joint_pos[:, 8:] = 0.0  # gripper mimic joints
        joint_pos[:, :7] = joints
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids=env_ids)
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step simulation without an action (used during resets only)."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def _visualize_markers(self):
        """Visualize markers for debugging."""
        if self.cfg.vis.vis_obs:
            self.held_asset_marker.visualize(self.held_pos_obs_frame + self.scene.env_origins, self.held_quat)
            self.fixed_asset_marker.visualize(self.fixed_pos_obs_frame + self.scene.env_origins, self.fixed_quat)