"""Run combinatorial pilot x copilot evaluation and save results table.

Usage (orchestrator — runs all 25 combinations as subprocesses):
    python scripts/sim_exp.py --task GearMesh

Single combination (worker — called by orchestrator, or manually):
    python scripts/sim_exp.py --task GearMesh --pilot kNNPilot --copilot ResidualCopilot --headless
"""

import argparse
import json
import numpy as np
import os
import sys

# ---------- constants (available before AppLauncher) ----------
PILOTS = ["LaggyPilot", "NoisyPilot", "ExpertPilot", "BCPilot", "kNNPilot"]

COPILOTS = ["GuidedDiffusionExpert", "GuidedDiffusionBC"]


def _results_dir(task):
    return os.path.join("logs", "sim_exp", task)


def _combo_path(task, pilot, copilot):
    return os.path.join(_results_dir(task), f"{pilot}__{copilot}.json")


# ---------- pre-parse to decide mode ----------
# If --pilot is present we are in worker mode and need AppLauncher.
# Otherwise we are the orchestrator and must NOT import Isaac Sim.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--pilot", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
_WORKER_MODE = _pre_args.pilot is not None


# ============================================================
# Worker mode — top-level AppLauncher setup (like play.py)
# ============================================================
if _WORKER_MODE:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="sim_exp worker: evaluate one pilot x copilot combination.")
    parser.add_argument("--task", type=str, required=True, choices=["GearMesh", "PegInsert", "NutThread"])
    parser.add_argument("--num_episodes", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--pilot", type=str, required=True,
                        choices=["LaggyPilot", "NoisyPilot", "ExpertPilot", "BCPilot", "kNNPilot"])
    parser.add_argument("--copilot", type=str, required=True,
                        choices=["None", "GuidedDiffusionBC", "GuidedDiffusionExpert", "ResidualBC", "ResidualCopilot"])
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        headless=True,
        kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error",
    )
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # post-AppLauncher imports
    import math
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import residual_copilot  # noqa: F401
    from residual_copilot.utils.constants import PILOT_NAME_MAP

    COPILOT_NAME_MAP = {
        "GuidedDiffusionBC":     ("GuidedDiffusion", f"shared_autonomy_policies/bc_teleop/{args_cli.task}_bc_teleop"),
        "GuidedDiffusionExpert": ("GuidedDiffusion", f"shared_autonomy_policies/bc_expert/{args_cli.task}_bc_expert"),
        "ResidualBC":            ("Residual", f"shared_autonomy_policies/residual_copilot/{args_cli.task}_bc_teleop/nn/FactoryXarm.pth"),
        "ResidualCopilot":       ("Residual", f"shared_autonomy_policies/residual_copilot/{args_cli.task}_noisy_knn/nn/FactoryXarm.pth"),
    }

    CLIP_VALUES = {
        "peg_insert": 0.15,
        "gear_mesh": 0.15,
        "nut_thread": 90.0,
    }

    def _build_dp_obs(env_unwrapped):
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

    def _collect_episodes(env_unwrapped, step_fn, on_done_fn, num_episodes, clip_val, label):
        successes = []
        clipped_errors = []
        raw_errors = []
        episodes_done = 0
        global_step = 0

        while episodes_done < num_episodes and simulation_app.is_running():
            with torch.inference_mode():
                dones = step_fn()
                if dones.any():
                    on_done_fn(dones)
                    for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                        if episodes_done >= num_episodes:
                            break
                        successes.append(bool(env_unwrapped.ep_succeeded[i].item()))
                        raw_err = float(env_unwrapped.assembly_error[i].item())
                        clipped_errors.append(min(raw_err, clip_val))
                        raw_errors.append(raw_err)
                        episodes_done += 1
                global_step += 1
                print(
                    f"\r  [{label}] episodes: {episodes_done}/{num_episodes} (step {global_step})",
                    end="", flush=True,
                )
        print()
        return successes, clipped_errors, raw_errors

    def run_worker():
        task = args_cli.task
        pilot_name = args_cli.pilot
        copilot_name = args_cli.copilot
        num_episodes = args_cli.num_episodes
        num_envs = args_cli.num_envs

        pilot_type, pilot_model_key = PILOT_NAME_MAP[pilot_name]
        label = f"{pilot_name} x {copilot_name}"
        container = [None]

        if copilot_name == "None":
            task_id = f"XArm-{task}-Residual"

            @hydra_task_config(task_id, "rl_games_cfg_entry_point")
            def _run(env_cfg, agent_cfg):
                env_cfg.scene.num_envs = num_envs
                env_cfg.pilot_model = pilot_model_key
                env_cfg.pilot_type = pilot_type

                env = gym.make(task_id, cfg=env_cfg)
                env_u = env.unwrapped
                clip_val = CLIP_VALUES[env_u.cfg_task.name]

                zero_actions = torch.zeros(
                    (env_u.num_envs, env_u.cfg.action_space), device=env_u.device,
                )
                env.reset()

                def step_fn():
                    _obs, _rew, terminated, truncated, _info = env.step(zero_actions)
                    return terminated | truncated

                container[0] = _collect_episodes(
                    env_u, step_fn, lambda _d: None, num_episodes, clip_val, label,
                )
                env.close()

            _run()

        else:
            method, hf_path = COPILOT_NAME_MAP[copilot_name]
            from residual_copilot.utils.utils import resolve_hf
            from residual_copilot.xarm_assembly_env.assembly_tasks_cfg import HF_MODELS_REPO
            resume_path = resolve_hf(HF_MODELS_REPO, hf_path, repo_type="model")

            if method == "GuidedDiffusion":
                from residual_copilot.pilot_models.bc_pilot import BC_Pilot

                task_id = f"XArm-{task}-GuidedDiffusion"

                @hydra_task_config(task_id, "rl_games_cfg_entry_point")
                def _run(env_cfg, agent_cfg):
                    env_cfg.scene.num_envs = num_envs
                    env_cfg.pilot_model = pilot_model_key
                    env_cfg.pilot_type = pilot_type

                    env = gym.make(task_id, cfg=env_cfg)
                    env_u = env.unwrapped
                    clip_val = CLIP_VALUES[env_u.cfg_task.name]

                    policy = BC_Pilot(resume_path)
                    env.reset()

                    def step_fn():
                        dp_obs = _build_dp_obs(env_u)
                        actions = policy.act(dp_obs, ref_action=env_u.base_actions)
                        _obs, _rew, terminated, truncated, _info = env.step(actions)
                        return terminated | truncated

                    def on_done_fn(dones):
                        if dones.any():
                            policy.reset()

                    container[0] = _collect_episodes(
                        env_u, step_fn, on_done_fn, num_episodes, clip_val, label,
                    )
                    env.close()

                _run()

            else:
                from rl_games.common import env_configurations, vecenv
                from rl_games.common.player import BasePlayer
                from rl_games.torch_runner import Runner
                from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

                task_id = f"XArm-{task}-Residual"

                @hydra_task_config(task_id, "rl_games_cfg_entry_point")
                def _run(env_cfg, agent_cfg):
                    env_cfg.scene.num_envs = num_envs
                    env_cfg.pilot_model = pilot_model_key
                    env_cfg.pilot_type = pilot_type

                    env = gym.make(task_id, cfg=env_cfg)
                    env_u = env.unwrapped
                    clip_val = CLIP_VALUES[env_u.cfg_task.name]

                    rl_device = agent_cfg["params"]["config"]["device"]
                    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
                    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
                    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
                    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

                    env = RlGamesVecEnvWrapper(
                        env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups,
                    )

                    vecenv.register(
                        "IsaacRlgWrapper",
                        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
                            config_name, num_actors, **kwargs
                        ),
                    )
                    env_configurations.register(
                        "rlgpu",
                        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
                    )

                    agent_cfg["params"]["load_checkpoint"] = True
                    agent_cfg["params"]["load_path"] = resume_path
                    agent_cfg["params"]["config"]["num_actors"] = env_u.num_envs

                    runner = Runner()
                    runner.load(agent_cfg)
                    agent: BasePlayer = runner.create_player()
                    agent.restore(resume_path)
                    agent.reset()

                    obs = env.reset()
                    if isinstance(obs, dict):
                        obs = obs["obs"]
                    _ = agent.get_batch_size(obs, 1)
                    if agent.is_rnn:
                        agent.init_rnn()

                    state = [obs]

                    def step_fn():
                        obs = state[0]
                        obs = agent.obs_to_torch(obs)
                        actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                        obs, _rew, dones, _info = env.step(actions)
                        state[0] = obs
                        return dones.bool() if isinstance(dones, torch.Tensor) else torch.tensor(
                            dones, dtype=torch.bool, device=env_u.device
                        )

                    container[0] = _collect_episodes(
                        env_u, step_fn, lambda _d: None, num_episodes, clip_val, label,
                    )
                    env.close()

                _run()

        # Save per-combination result
        successes, clipped_errors, raw_errors = container[0]
        result = {
            "success_mean": float(np.mean(successes)),
            "success_std": float(np.std(successes)),
            "error_mean": float(np.mean(clipped_errors)),
            "error_std": float(np.std(clipped_errors)),
            "raw_error_mean": float(np.mean(raw_errors)),
            "raw_error_std": float(np.std(raw_errors)),
        }
        os.makedirs(_results_dir(task), exist_ok=True)
        out_file = _combo_path(task, pilot_name, copilot_name)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved {out_file}")


# ============================================================
# Orchestrator mode — no Isaac Sim, spawns subprocesses
# ============================================================
def _print_table(task, num_episodes):
    """Load per-combination JSONs and print/save merged table."""
    results = {}
    for copilot in COPILOTS:
        for pilot in PILOTS:
            path = _combo_path(task, pilot, copilot)
            if os.path.exists(path):
                with open(path) as f:
                    results[(copilot, pilot)] = json.load(f)

    if not results:
        return

    merged_path = os.path.join(_results_dir(task), "results.json")
    data = {"task": task, "num_episodes": num_episodes, "results": {}}
    for copilot in COPILOTS:
        copilot_data = {}
        for pilot in PILOTS:
            if (copilot, pilot) in results:
                copilot_data[pilot] = results[(copilot, pilot)]
        if copilot_data:
            data["results"][copilot] = copilot_data
    with open(merged_path, "w") as f:
        json.dump(data, f, indent=2)

    col_w = 18
    pending = "---".center(col_w)
    for metric_name, mean_key, std_key in [
        ("Success Rate", "success_mean", "success_std"),
        ("Clipped Assembly Error", "error_mean", "error_std"),
    ]:
        print(f"\n=== {metric_name} (mean +/- std) ===")
        header = f"{'':>26s}" + "".join(f"{p:>{col_w}s}" for p in PILOTS)
        print(header)
        for copilot in COPILOTS:
            row = f"{copilot:>26s}"
            for pilot in PILOTS:
                if (copilot, pilot) in results:
                    m = results[(copilot, pilot)][mean_key]
                    s = results[(copilot, pilot)][std_key]
                    row += f"{m:.3f}+/-{s:.3f}".rjust(col_w)
                else:
                    row += pending
            print(row)

    print(f"\n[INFO] Merged results saved to {merged_path}")


def _fmt_elapsed(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def orchestrate():
    import subprocess
    import time

    parser = argparse.ArgumentParser(description="sim_exp orchestrator: run all combinations.")
    parser.add_argument("--task", type=str, required=True, choices=["GearMesh", "PegInsert", "NutThread"])
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--headless", action="store_true", default=True, help="(ignored, workers are always headless)")
    args, _ = parser.parse_known_args()

    os.makedirs(_results_dir(args.task), exist_ok=True)
    t0 = time.time()
    combo_idx = 0
    total_combos = len(COPILOTS) * len(PILOTS)

    for copilot_name in COPILOTS:
        for pilot_name in PILOTS:
            combo_idx += 1
            elapsed = _fmt_elapsed(time.time() - t0)
            out_file = _combo_path(args.task, pilot_name, copilot_name)
            if os.path.exists(out_file):
                print(f"[SKIP] [{elapsed}] ({combo_idx}/{total_combos}) {pilot_name} x {copilot_name} — already done")
                continue

            print(f"\n{'=' * 60}")
            print(f"  [{elapsed}] ({combo_idx}/{total_combos}) Launching: pilot={pilot_name}, copilot={copilot_name}")
            print(f"{'=' * 60}")

            cmd = [
                sys.executable, __file__,
                "--task", args.task,
                "--pilot", pilot_name,
                "--copilot", copilot_name,
                "--num_episodes", str(args.num_episodes),
                "--num_envs", str(args.num_envs),
                "--headless",
            ]
            result = subprocess.run(cmd)
            elapsed = _fmt_elapsed(time.time() - t0)
            if result.returncode != 0:
                print(f"[ERROR] [{elapsed}] {pilot_name} x {copilot_name} failed (exit {result.returncode})")
            else:
                print(f"[DONE] [{elapsed}] {pilot_name} x {copilot_name}")

            _print_table(args.task, args.num_episodes)

    elapsed = _fmt_elapsed(time.time() - t0)
    print(f"\n[INFO] All combinations finished. Total time: {elapsed}")
    _print_table(args.task, args.num_episodes)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    if _WORKER_MODE:
        run_worker()
        simulation_app.close()
    else:
        orchestrate()
