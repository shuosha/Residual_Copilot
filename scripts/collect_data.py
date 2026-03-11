"""Collect imitation learning data by running a pilot model and saving successful episodes."""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect IL data using kNNPilot + ResidualCopilot.")
parser.add_argument("--task", type=str, required=True, choices=["GearMesh", "PegInsert", "NutThread"])
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, required=True, help="Number of successful episodes to collect.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save collected data. Defaults to logs/data/<task>_ResidualCopilot_<num_episodes>ep.")
parser.add_argument("--no_images", dest="save_images", action="store_false", help="Disable saving RGB images.")
parser.set_defaults(save_images=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import time
import torch
from pathlib import Path
from tqdm import tqdm

import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config

import residual_copilot  # noqa: F401

from residual_copilot.utils.constants import PILOT_NAME_MAP


class DataCollector:
    """
    Buffers obs/action per env in RAM; flushes to disk only on successful episode end.

    Layout::

        output_dir/
          episodes_played/episode_{id:04d}/camera_0/rgb/{t:06d}.jpg   (all played, if save_images)
          episodes_collected/episode_{id:04d}/robot/{t:06d}.json       (successful only)
          meta.jsonl                                                    (one line per played episode)
    """

    def __init__(self, output_dir: str, num_envs: int, save_images: bool = True):
        self.output_dir = Path(output_dir)
        self.num_envs = num_envs
        self.save_images = save_images

        self.played_root = self.output_dir / "episodes_played"
        self.collected_root = self.output_dir / "episodes_collected"
        self.played_root.mkdir(parents=True, exist_ok=True)
        self.collected_root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "meta.jsonl"

        self.episodes_played = 0
        self.episodes_collected = 0

        self.curr_played_id = np.full((num_envs,), -1, dtype=np.int64)
        self.buffers = [[] for _ in range(num_envs)]
        self.t_in_ep = np.zeros((num_envs,), dtype=np.int64)

        for env_id in range(num_envs):
            self._start_new_played_episode(env_id)

    def _played_dir(self, played_id: int) -> Path:
        return self.played_root / f"episode_{played_id:04d}"

    def _collected_dir(self, collected_id: int) -> Path:
        return self.collected_root / f"episode_{collected_id:04d}"

    def _start_new_played_episode(self, env_id: int):
        played_id = self.episodes_played
        self.episodes_played += 1
        self.curr_played_id[env_id] = played_id
        self.buffers[env_id] = []
        self.t_in_ep[env_id] = 0
        if self.save_images:
            (self._played_dir(played_id) / "camera_0" / "rgb").mkdir(parents=True, exist_ok=True)

    def save_step(self, env_id: int, obs_vec: np.ndarray, action_vec: np.ndarray, img_bgr=None):
        played_id = int(self.curr_played_id[env_id])
        t = int(self.t_in_ep[env_id])

        if self.save_images and img_bgr is not None:
            img_path = self._played_dir(played_id) / "camera_0" / "rgb" / f"{t:06d}.jpg"
            cv2.imwrite(str(img_path), img_bgr)

        self.buffers[env_id].append({
            "obs": np.asarray(obs_vec).reshape(-1).astype(np.float32, copy=False),
            "action": np.asarray(action_vec).reshape(-1).astype(np.float32, copy=False),
        })
        self.t_in_ep[env_id] += 1

    def end_episode(self, env_id: int, success: bool):
        played_id = int(self.curr_played_id[env_id])
        num_steps = int(self.t_in_ep[env_id])
        collected_id = None

        if success:
            collected_id = self.episodes_collected
            self.episodes_collected += 1
            robot_dir = self._collected_dir(collected_id) / "robot"
            robot_dir.mkdir(parents=True, exist_ok=True)
            for t, entry in enumerate(self.buffers[env_id]):
                with open(robot_dir / f"{t:06d}.json", "w") as f:
                    json.dump({
                        "obs": entry["obs"].tolist(),
                        "action": entry["action"].tolist(),
                    }, f, indent=2)

        meta = {
            "episode_played": played_id,
            "env_id": int(env_id),
            "success": bool(success),
            "num_steps": num_steps,
            "episode_collected": collected_id,
        }
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self._start_new_played_episode(env_id)


COPILOT_HF_PATH = "shared_autonomy_policies/residual_copilot/{task}_noisy_knn/nn/FactoryXarm.pth"


def main():
    # Always use kNNPilot + ResidualCopilot.
    pilot_type, pilot_model_key = PILOT_NAME_MAP["kNNPilot"]
    task_id = f"XArm-{args_cli.task}-Residual"

    # Build default output directory if not specified.
    if args_cli.output_dir is None:
        args_cli.output_dir = os.path.join(
            "logs", "data",
            f"{args_cli.task}_ResidualCopilot_{args_cli.num_episodes}ep",
        )

    # Resolve copilot checkpoint from HuggingFace.
    from residual_copilot.utils.utils import resolve_hf
    from residual_copilot.xarm_assembly_env.assembly_tasks_cfg import HF_MODELS_REPO

    hf_path = COPILOT_HF_PATH.format(task=args_cli.task)
    resume_path = resolve_hf(HF_MODELS_REPO, hf_path, repo_type="model")

    @hydra_task_config(task_id, "rl_games_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.pilot_model = pilot_model_key
        env_cfg.pilot_type = pilot_type
        env_cfg.vis.store_rgb = args_cli.save_images

        env = gym.make(task_id, cfg=env_cfg)
        env_unwrapped = env.unwrapped
        num_envs = env_unwrapped.num_envs

        # Load RL-Games copilot agent.
        import math
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        rl_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env})

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = num_envs

        runner = Runner()
        runner.load(agent_cfg)
        rl_agent = runner.create_player()
        rl_agent.restore(resume_path)
        rl_agent.reset()

        # Initialize agent with first obs.
        obs = rl_env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = rl_agent.get_batch_size(obs, 1)
        if rl_agent.is_rnn:
            rl_agent.init_rnn()

        collector = DataCollector(args_cli.output_dir, num_envs, save_images=args_cli.save_images)

        # Save run metadata
        infos_dir = Path(args_cli.output_dir) / "infos"
        infos_dir.mkdir(parents=True, exist_ok=True)
        with open(infos_dir / "infos.json", "w") as f:
            json.dump({
                "task": args_cli.task,
                "pilot": "kNNPilot",
                "copilot": "ResidualCopilot",
                "pilot_model": pilot_model_key,
                "pilot_type": pilot_type,
                "num_episodes": args_cli.num_episodes,
            }, f, indent=2)

        max_episodes = args_cli.num_episodes
        print(f"[INFO] Collecting {max_episodes} successful episodes with kNNPilot + ResidualCopilot on {args_cli.task}.\n")
        pbar = tqdm(total=max_episodes, desc="Episodes collected", unit="ep")

        while simulation_app.is_running() and collector.episodes_collected < max_episodes:
            with torch.inference_mode():
                # Snapshot obs before stepping (first 20 dims = low-level obs).
                obs_snapshot = obs.clone() if not isinstance(obs, dict) else obs["obs"].clone()

                # Step the environment.
                obs_t = rl_agent.obs_to_torch(obs)
                actions = rl_agent.get_action(obs_t, is_deterministic=rl_agent.is_deterministic)
                obs, _rew, dones, _info = rl_env.step(actions)

                obs_snap_np = obs_snapshot[:, :20].detach().cpu().numpy()
                env_act_np = env_unwrapped.env_actions.detach().cpu().numpy()
                ep_succeeded_np = env_unwrapped.ep_succeeded.detach().cpu().numpy()
                dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)

                if args_cli.save_images:
                    img_np_all = env_unwrapped.front_rgb.detach().cpu().numpy()

                prev_collected = collector.episodes_collected
                for env_id in range(num_envs):
                    obs_np = obs_snap_np[env_id].reshape(-1)
                    act_np = env_act_np[env_id].reshape(-1)
                    img_bgr = None
                    if args_cli.save_images:
                        img_bgr = cv2.cvtColor(img_np_all[env_id], cv2.COLOR_RGB2BGR)

                    collector.save_step(env_id, obs_np, act_np, img_bgr)

                    if bool(dones_np[env_id]):
                        success = bool(ep_succeeded_np[env_id])
                        collector.end_episode(env_id, success)

                if collector.episodes_collected > prev_collected:
                    pbar.update(collector.episodes_collected - prev_collected)

                if collector.episodes_collected >= max_episodes:
                    break

        pbar.close()
        print(f"[INFO] Collected {collector.episodes_collected} successful episodes → {args_cli.output_dir}")
        print(f"[INFO] Total episodes played: {collector.episodes_played}")
        env.close()

    _run()


if __name__ == "__main__":
    main()
    simulation_app.close()
