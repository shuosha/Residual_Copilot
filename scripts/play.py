"""Play a pilot model, optionally with a copilot (residual RL) on top."""

import argparse
import cv2
import json
import numpy as np
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a pilot model, optionally with a copilot on top.")
parser.add_argument("--task", type=str, required=True, choices=["GearMesh", "PegInsert", "NutThread"])
parser.add_argument("--pilot", type=str, required=True,
                    choices=["LaggyPilot", "NoisyPilot", "ExpertPilot", "BCPilot", "kNNPilot", "ReplayPilot"],
                    help="Pilot (base) model to run.")
parser.add_argument("--copilot", type=str, default=None,
                    choices=["GuidedDiffusionBC", "GuidedDiffusionExpert", "ResidualBC", "ResidualCopilot"],
                    help="Copilot (residual RL) model to load. If omitted, runs pilot-only with zero residual.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--record", action="store_true", default=False,
                    help="Record rollouts: save episode stats and RGB images to logs/rollouts/.")
parser.add_argument("--no_rand", action="store_true", default=False, help="Disable domain randomization (keep sim deterministic).")
AppLauncher.add_app_launcher_args(parser)
# suppress verbose Kit/USD logs by default
parser.set_defaults(kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.record:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

import residual_copilot  # noqa: F401

# ANSI color codes
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

from residual_copilot.utils.constants import PILOT_NAME_MAP

COPILOT_NAME_MAP = {
    "GuidedDiffusionBC":     ("GuidedDiffusion", f"shared_autonomy_policies/bc_teleop/{args_cli.task}_bc_teleop"),
    "GuidedDiffusionExpert": ("GuidedDiffusion", f"shared_autonomy_policies/bc_expert/{args_cli.task}_bc_expert"),
    "ResidualBC":            ("Residual",        f"shared_autonomy_policies/residual_copilot/{args_cli.task}_bc_teleop/nn/FactoryXarm.pth"),
    "ResidualCopilot":       ("Residual",        f"shared_autonomy_policies/residual_copilot/{args_cli.task}_noisy_knn/nn/FactoryXarm.pth"),
}


def _print_run_info(task, pilot_model, copilot_model, num_envs):
    copilot_str = copilot_model if copilot_model else "none (pilot-only)"
    num_envs_str = str(num_envs) if num_envs is not None else "default"
    border = _BOLD + _CYAN + "=" * 52 + _RESET
    print(border)
    print(f"{_BOLD}{_CYAN}{'  RESIDUAL COPILOT — RUN CONFIG':^52}{_RESET}")
    print(border)
    print(f"  {_BOLD}Task       :{_RESET}  {_GREEN}{task}{_RESET}")
    print(f"  {_BOLD}Pilot      :{_RESET}  {_GREEN}{pilot_model}{_RESET}")
    print(f"  {_BOLD}Copilot    :{_RESET}  {_YELLOW}{copilot_str}{_RESET}")
    print(f"  {_BOLD}Num envs   :{_RESET}  {num_envs_str}")
    print(border)


def _make_rollout_dir():
    """Create and return the rollout output directory path.

    If the directory already exists, prompt the user for confirmation before
    overwriting.  Exits if the user declines.
    """
    task = args_cli.task
    pilot = args_cli.pilot
    copilot = args_cli.copilot
    if copilot is not None:
        dir_name = f"eval_{task}_with_{copilot}_and_{pilot}"
    else:
        dir_name = f"eval_{task}_with_{pilot}"
    
    if args_cli.no_rand:
        dir_name += "_no_rand"
    rollout_path = os.path.abspath(os.path.join("logs", "rollouts", dir_name))

    if os.path.exists(rollout_path):
        print(f"{_YELLOW}[WARNING]{_RESET} Rollout directory already exists:\n  {rollout_path}")
        answer = input("Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborting.")
            sys.exit(0)
        import shutil
        shutil.rmtree(rollout_path)

    os.makedirs(rollout_path, exist_ok=True)
    return rollout_path


def _save_rollout_meta(rollout_path, ep_stats):
    """Write meta/infos.json and meta/stats.json into rollout_path."""
    meta_dir = os.path.join(rollout_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    infos = {
        "task": args_cli.task,
        "pilot": args_cli.pilot,
        "copilot": args_cli.copilot,
        "num_envs": args_cli.num_envs,
    }
    with open(os.path.join(meta_dir, "infos.json"), "w") as f:
        json.dump(infos, f, indent=2)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(ep_stats, f, indent=2)

    successes = sum(1 for v in ep_stats.values() if v["success"])
    total = len(ep_stats)
    print(f"[INFO] Saved rollout meta to: {meta_dir}")
    print(f"[INFO] Success rate: {successes}/{total}")


def _format_timestep(obs, base_actions, env_actions, qpos):
    """Build a structured dict for one env at one timestep.

    Args:
        obs: flat policy observation (35D) for this env.
        base_actions: base_actions (8D) for this env.
        env_actions: env_actions (8D) for this env.
        qpos: qpos_targets (7D) for this env.

    Obs layout (from obs_order + prev_actions, total 35D):
        fingertip_pos          [0:3]
        fingertip_quat         [3:7]
        gripper                [7:8]
        fingertip_pos_rel_fixed [8:11]
        fingertip_pos_rel_held  [11:14]
        ee_linvel_fd           [14:17]
        ee_angvel_fd           [17:20]
        base_fingertip_pos     [20:23]   (from obs; may differ from base_actions
        base_fingertip_quat    [23:27]    after noise/lag augmentation is applied
        base_gripper           [27:28]    within the same step)
        prev_actions           [28:35]
    """
    return {
        "obs.fingertip_pos":          obs[0:3].tolist(),
        "obs.fingertip_quat":         obs[3:7].tolist(),
        "obs.gripper":                obs[7:8].tolist(),
        "obs.fingertip_pos_rel_fixed": obs[8:11].tolist(),
        "obs.fingertip_pos_rel_held": obs[11:14].tolist(),
        "obs.ee_linvel_fd":           obs[14:17].tolist(),
        "obs.ee_angvel_fd":           obs[17:20].tolist(),
        "obs.qpos":                   qpos.tolist(),
        "base_action.fingertip_pos":  base_actions[0:3].tolist(),
        "base_action.fingertip_quat": base_actions[3:7].tolist(),
        "base_action.gripper":        base_actions[7:8].tolist(),
        "action.fingertip_pos":       env_actions[0:3].tolist(),
        "action.fingertip_quat":      env_actions[3:7].tolist(),
        "action.gripper":             env_actions[7:8].tolist(),
        "action.qpos":               qpos.tolist(),
    }


def _build_dp_obs(env_unwrapped):
    """Build LeRobot-format obs dict from env tensors for the diffusion policy."""
    return {
        "observation.state": torch.cat([
            env_unwrapped.fingertip_midpoint_pos,
            env_unwrapped.fingertip_midpoint_quat,
            env_unwrapped.gripper,
            env_unwrapped.ee_linvel_fd,
            env_unwrapped.ee_angvel_fd,
        ], dim=-1),
        "observation.environment_state": torch.cat([
            env_unwrapped.fingertip_midpoint_pos - env_unwrapped.fixed_pos_obs_frame,
            env_unwrapped.fingertip_midpoint_pos - env_unwrapped.held_pos_obs_frame,
        ], dim=-1),
    }


def _run_loop_guided_diffusion(env, env_unwrapped, policy):
    """Run guided diffusion eval loop (no recording)."""
    num_envs = env_unwrapped.num_envs
    obs, _ = env.reset()
    episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env_unwrapped.device)
    global_step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            dp_obs = _build_dp_obs(env_unwrapped)
            base_actions = env_unwrapped.base_actions
            actions = policy.act(dp_obs, ref_action=base_actions)

            obs, _rew, terminated, truncated, _info = env.step(actions)
            dones = terminated | truncated

            if dones.any():
                policy.reset()

            episode_done |= dones.bool()
            n_done = int(episode_done.sum().item())
            print(f"\r[Step {global_step:5d}]  done: {n_done}/{num_envs}  ongoing: {num_envs - n_done}/{num_envs}", end="", flush=True)
            global_step += 1

            if torch.all(episode_done):
                print()
                print("[INFO] All episodes completed.")
                break


def _run_loop_recording(env_unwrapped, step_fn, on_done_fn, rollout_path):
    """Shared recording loop for all copilot types.

    Args:
        env_unwrapped: unwrapped DirectRLEnv with .num_envs, .base_actions, etc.
        step_fn: callable() -> (obs_snapshot_np, dones_tensor) — advance one step and
                 return the observation snapshot (numpy, shape [N, obs_dim]) and dones.
        on_done_fn: callable(dones) — called after each step with the dones tensor
                    (e.g. to reset a diffusion policy).
        rollout_path: directory path to write episode data into.

    Returns:
        ep_stats dict suitable for _save_rollout_meta().
    """
    num_envs = env_unwrapped.num_envs
    episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env_unwrapped.device)
    success_ts = [-1] * num_envs
    timesteps = [0] * num_envs
    global_step = 0

    store_rgb = env_unwrapped.cfg.vis.store_rgb

    robot_buffers = [[] for _ in range(num_envs)]
    for env_id in range(num_envs):
        if store_rgb:
            os.makedirs(os.path.join(rollout_path, f"episode_{env_id:04d}", "camera_0", "rgb"), exist_ok=True)
        os.makedirs(os.path.join(rollout_path, f"episode_{env_id:04d}", "robot"), exist_ok=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            obs_snap_np, dones = step_fn()

            base_act_np = env_unwrapped.base_actions.detach().cpu().numpy()
            env_act_np = env_unwrapped.env_actions.detach().cpu().numpy()
            qpos_np = env_unwrapped.qpos_targets.detach().cpu().numpy()
            img_np_all = env_unwrapped.front_rgb.cpu().numpy() if store_rgb else None

            for env_id in range(num_envs):
                if episode_done[env_id]:
                    continue

                done = bool(dones[env_id].item())
                t = timesteps[env_id]

                robot_buffers[env_id].append(_format_timestep(
                    obs_snap_np[env_id],
                    base_act_np[env_id],
                    env_act_np[env_id],
                    qpos_np[env_id],
                ))

                if store_rgb and img_np_all is not None:
                    rgb_dir = os.path.join(rollout_path, f"episode_{env_id:04d}", "camera_0", "rgb")
                    img_bgr = cv2.cvtColor(img_np_all[env_id], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(rgb_dir, f"{t:06d}.jpg"), img_bgr)

                if done:
                    ep_succeeded = bool(env_unwrapped.ep_succeeded[env_id].item())
                    if ep_succeeded and success_ts[env_id] == -1:
                        success_ts[env_id] = t
                    episode_done[env_id] = True

                    robot_dir = os.path.join(rollout_path, f"episode_{env_id:04d}", "robot")
                    for step_t, entry in enumerate(robot_buffers[env_id]):
                        with open(os.path.join(robot_dir, f"{step_t:06d}.json"), "w") as f:
                            json.dump(entry, f, indent=2)

                    # print(
                    #     f"[INFO] Episode {env_id} done at step {t}, succeeded: {ep_succeeded}"
                    # )
                else:
                    timesteps[env_id] += 1

            on_done_fn(dones)

        n_done = int(episode_done.sum().item())
        print(f"\r[Step {global_step:5d}]  done: {n_done}/{num_envs}  ongoing: {num_envs - n_done}/{num_envs}", end="", flush=True)
        global_step += 1

        if torch.all(episode_done):
            print()
            print("[INFO] All episodes completed.")
            break

    return {
        f"episode_{env_id:04d}": {
            "success": success_ts[env_id] != -1,
            "success_timestep": success_ts[env_id],
        }
        for env_id in range(num_envs)
    }


def _run_loop_plain(env, env_unwrapped, agent):
    """Run the eval loop without recording."""
    num_envs = env_unwrapped.num_envs
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env_unwrapped.device)
    global_step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _rew, dones, _info = env.step(actions)

            episode_done |= dones.bool() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.bool, device=env_unwrapped.device)
            n_done = int(episode_done.sum().item())
            print(f"\r[Step {global_step:5d}]  done: {n_done}/{num_envs}  ongoing: {num_envs - n_done}/{num_envs}", end="", flush=True)
            global_step += 1


def main():
    _print_run_info(args_cli.task, args_cli.pilot, args_cli.copilot, args_cli.num_envs)

    pilot_type, pilot_model_key = PILOT_NAME_MAP[args_cli.pilot]

    rollout_path = _make_rollout_dir() if args_cli.record else None

    if args_cli.copilot is not None:
        # --- copilot mode ---
        from residual_copilot.utils.utils import resolve_hf
        from residual_copilot.xarm_assembly_env.assembly_tasks_cfg import HF_MODELS_REPO

        method, hf_path = COPILOT_NAME_MAP[args_cli.copilot]
        task_id = f"XArm-{args_cli.task}-{method}"
        resume_path = resolve_hf(HF_MODELS_REPO, hf_path, repo_type="model")

        if method == "GuidedDiffusion":
            # --- guided diffusion copilot (DiffusionPolicy, not RL-Games) ---
            from residual_copilot.pilot_models.bc_pilot import BC_Pilot

            @hydra_task_config(task_id, "rl_games_cfg_entry_point")
            def _run(env_cfg, agent_cfg):
                if args_cli.num_envs is not None:
                    env_cfg.scene.num_envs = args_cli.num_envs

                env_cfg.pilot_model = pilot_model_key
                env_cfg.pilot_type = pilot_type

                if args_cli.record:
                    env_cfg.vis.store_rgb = True

                if args_cli.no_rand:
                    env_cfg.dmr.rand_ctrl = False
                    env_cfg.dmr.aug_data = False
                    env_cfg.vis.order_envs = True

                env = gym.make(task_id, cfg=env_cfg)
                env.unwrapped.cfg_task.success_rotation_threshold_deg = 180.0

                policy = BC_Pilot(resume_path)

                if args_cli.record:
                    env_u = env.unwrapped
                    state = [env.reset()[0]]  # state[0] = obs (mutable box)

                    def step_fn():
                        obs = state[0]
                        obs_policy = obs["policy"] if isinstance(obs, dict) else obs
                        obs_snap = obs_policy.detach().cpu().numpy()
                        dp_obs = _build_dp_obs(env_u)
                        actions = policy.act(dp_obs, ref_action=env_u.base_actions)
                        obs, _rew, terminated, truncated, _info = env.step(actions)
                        state[0] = obs
                        return obs_snap, terminated | truncated

                    def on_done_fn(dones):
                        if dones.any():
                            policy.reset()

                    ep_stats = _run_loop_recording(env_u, step_fn, on_done_fn, rollout_path)
                    _save_rollout_meta(rollout_path, ep_stats)
                else:
                    _run_loop_guided_diffusion(env, env.unwrapped, policy)

                env.close()

            _run()

        else:
            # --- RL-Games copilot (ResidualBC, ResidualCopilot) ---
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.player import BasePlayer
            from rl_games.torch_runner import Runner

            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

            @hydra_task_config(task_id, "rl_games_cfg_entry_point")
            def _run(env_cfg, agent_cfg):
                if args_cli.num_envs is not None:
                    env_cfg.scene.num_envs = args_cli.num_envs

                env_cfg.pilot_model = pilot_model_key
                env_cfg.pilot_type = pilot_type

                if args_cli.record:
                    env_cfg.vis.store_rgb = True

                if args_cli.no_rand:
                    env_cfg.dmr.rand_ctrl = False
                    env_cfg.dmr.aug_data = False
                    env_cfg.vis.order_envs = True

                env = gym.make(task_id, cfg=env_cfg)
                env.unwrapped.cfg_task.success_rotation_threshold_deg = 180.0

                rl_device = agent_cfg["params"]["config"]["device"]
                clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
                clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
                obs_groups = agent_cfg["params"]["env"].get("obs_groups")
                concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

                env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

                vecenv.register(
                    "IsaacRlgWrapper",
                    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
                )
                env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

                agent_cfg["params"]["load_checkpoint"] = True
                agent_cfg["params"]["load_path"] = resume_path
                agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

                runner = Runner()
                runner.load(agent_cfg)
                agent: BasePlayer = runner.create_player()
                agent.restore(resume_path)
                agent.reset()

                if args_cli.record:
                    obs = env.reset()
                    if isinstance(obs, dict):
                        obs = obs["obs"]
                    _ = agent.get_batch_size(obs, 1)
                    if agent.is_rnn:
                        agent.init_rnn()
                    env_u = env.unwrapped
                    state = [obs]  # state[0] = obs (mutable box)

                    def step_fn():
                        obs = state[0]
                        obs_snap = obs.clone() if not isinstance(obs, dict) else obs["obs"].clone()
                        obs = agent.obs_to_torch(obs)
                        actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                        obs, _rew, dones, _info = env.step(actions)
                        state[0] = obs
                        return obs_snap.detach().cpu().numpy(), dones

                    ep_stats = _run_loop_recording(env_u, step_fn, lambda _: None, rollout_path)
                    _save_rollout_meta(rollout_path, ep_stats)
                else:
                    _run_loop_plain(env, env.unwrapped, agent)

                env.close()

            _run()

    else:
        # --- pilot-only mode (zero residual) ---
        task_id = f"XArm-{args_cli.task}-Residual"

        @hydra_task_config(task_id, "rl_games_cfg_entry_point")
        def _run(env_cfg, agent_cfg):
            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = args_cli.num_envs

            env_cfg.pilot_model = pilot_model_key
            env_cfg.pilot_type = pilot_type

            if args_cli.record:
                env_cfg.vis.store_rgb = True

            if args_cli.no_rand:
                env_cfg.dmr.rand_ctrl = False
                env_cfg.dmr.aug_data = False
                env_cfg.vis.order_envs = True

            env = gym.make(task_id, cfg=env_cfg)
            env.unwrapped.cfg_task.success_rotation_threshold_deg = 180.0

            zero_actions = torch.zeros(
                (env.unwrapped.num_envs, env.unwrapped.cfg.action_space),
                device=env.unwrapped.device,
            )

            num_envs = env.unwrapped.num_envs
            env_unwrapped = env.unwrapped

            if args_cli.record:
                store_rgb = env_unwrapped.cfg.vis.store_rgb
                episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env_unwrapped.device)
                success_ts = [-1] * num_envs
                timesteps = [0] * num_envs
                global_step = 0

                robot_buffers = [[] for _ in range(num_envs)]
                for env_id in range(num_envs):
                    if store_rgb:
                        os.makedirs(os.path.join(rollout_path, f"episode_{env_id:04d}", "camera_0", "rgb"), exist_ok=True)
                    os.makedirs(os.path.join(rollout_path, f"episode_{env_id:04d}", "robot"), exist_ok=True)

                obs, _ = env.reset()

                while simulation_app.is_running():
                    with torch.inference_mode():
                        obs, _rew, terminated, truncated, _info = env.step(zero_actions)
                        dones = terminated | truncated

                        obs_policy = obs["policy"] if isinstance(obs, dict) else obs
                        obs_np = obs_policy.detach().cpu().numpy()
                        base_act_np = env_unwrapped.base_actions.detach().cpu().numpy()
                        qpos_np = env_unwrapped.qpos_targets.detach().cpu().numpy()
                        if store_rgb:
                            img_tensor = env_unwrapped.front_rgb

                        for env_id in range(num_envs):
                            if episode_done[env_id]:
                                continue

                            done = dones[env_id].item()
                            t = timesteps[env_id]

                            o = obs_np[env_id]
                            ba = base_act_np[env_id]
                            robot_buffers[env_id].append({
                                "obs.fingertip_pos":           o[0:3].tolist(),
                                "obs.fingertip_quat":          o[3:7].tolist(),
                                "obs.gripper":                 o[7:8].tolist(),
                                "obs.fingertip_pos_rel_fixed": o[8:11].tolist(),
                                "obs.fingertip_pos_rel_held":  o[11:14].tolist(),
                                "obs.ee_linvel_fd":            o[14:17].tolist(),
                                "obs.ee_angvel_fd":            o[17:20].tolist(),
                                "obs.qpos":                    qpos_np[env_id].tolist(),
                                "base_action.fingertip_pos":   ba[0:3].tolist(),
                                "base_action.fingertip_quat":  ba[3:7].tolist(),
                                "base_action.gripper":         ba[7:8].tolist(),
                            })

                            if store_rgb:
                                rgb_dir = os.path.join(
                                    rollout_path, f"episode_{env_id:04d}", "camera_0", "rgb"
                                )
                                img = img_tensor[env_id].cpu().numpy()
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(rgb_dir, f"{t:06d}.jpg"), img_bgr)

                            if done:
                                ep_succeeded = bool(env_unwrapped.ep_succeeded[env_id].item())
                                if ep_succeeded and success_ts[env_id] == -1:
                                    success_ts[env_id] = t
                                episode_done[env_id] = True

                                robot_dir = os.path.join(rollout_path, f"episode_{env_id:04d}", "robot")
                                for step_t, entry in enumerate(robot_buffers[env_id]):
                                    with open(os.path.join(robot_dir, f"{step_t:06d}.json"), "w") as f:
                                        json.dump(entry, f, indent=2)

                                print(
                                    f"[INFO] Episode {env_id} done at step {t}, "
                                    f"succeeded: {ep_succeeded}"
                                )
                            else:
                                timesteps[env_id] += 1

                    n_done = int(episode_done.sum().item())
                    print(f"\r[Step {global_step:5d}]  done: {n_done}/{num_envs}  ongoing: {num_envs - n_done}/{num_envs}", end="", flush=True)
                    global_step += 1

                    if torch.all(episode_done):
                        print()
                        print("[INFO] All episodes completed.")
                        break

                ep_stats = {
                    f"episode_{env_id:04d}": {
                        "success": success_ts[env_id] != -1,
                        "success_timestep": success_ts[env_id],
                    }
                    for env_id in range(num_envs)
                }
                _save_rollout_meta(rollout_path, ep_stats)

            else:
                obs, _ = env.reset()
                episode_done = torch.zeros(num_envs, dtype=torch.bool, device=env_unwrapped.device)
                global_step = 0

                while simulation_app.is_running():
                    with torch.inference_mode():
                        obs, _rew, terminated, truncated, _info = env.step(zero_actions)
                        dones = terminated | truncated
                        episode_done |= dones.bool()
                        n_done = int(episode_done.sum().item())
                        print(f"\r[Step {global_step:5d}]  done: {n_done}/{num_envs}  ongoing: {num_envs - n_done}/{num_envs}", end="", flush=True)
                        global_step += 1

            env.close()

        _run()


if __name__ == "__main__":
    main()
    simulation_app.close()
