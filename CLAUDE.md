# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Installation

```bash
git clone --recurse-submodules <repo>
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv sync
```

The package manager is **uv** (not pip). The Python package is named `residual_copilot` and maps to the `source/` directory (see `pyproject.toml`).

Running any Isaac Lab environment requires an active **Isaac Sim 5.1.0** / Isaac Lab 2.3.2 runtime — `import residual_copilot` outside that runtime will fail on `pxr`/USD imports.

## Linting

```bash
flake8 source/ scripts/
```

Config in `.flake8`: max line length 120, max complexity 30, Google-style docstrings.

## Validation Scripts (no formal test suite)

```bash
python scripts/list_envs.py          # confirm environment registration
```

## Training

```bash
python scripts/train.py \
  --task XArm-GearMesh-Residual \
  --num_envs 128 \
  --headless
```

Key CLI args: `--task`, `--num_envs`, `--checkpoint` (resume), `--wandb_project`, `--distributed` (multi-GPU).

Evaluation/rendering:

```bash
python scripts/play.py --task GearMesh --pilot kNNPilot --copilot ResidualCopilot
```

## Architecture

### Control Stack

```
obs (11D) → Pilot (KNN or BC/DiffusionPolicy) → base_action (8D)
         ↓
[+noise or +lag augmentation if pilot_type != "none"]
         ↓
obs (35D policy / more for critic) → RL Residual Policy → residual_action (7D)
         ↓
_apply_residual() → admittance control (control.py) → IK (Pinocchio/SAPIEN) → joint targets
         ↓
Isaac Sim physics
```

### Observation Space
- **Policy obs (35D):** fingertip pos, quat, gripper, relative held/base positions, velocities, base actions — noisy (matches data distribution)
- **Critic state:** adds ground-truth joint positions, held object pose, base object pose — ground truth for critic-only

### Three Task Environments

Registered in `source/xarm_assembly_env/__init__.py`:
- `XArm-GearMesh-Residual`
- `XArm-NutThread-Residual`
- `XArm-PegInsert-Residual`

Each task defines its own asset configs, reward scales, and data/model HuggingFace paths in `assembly_tasks_cfg.py`.

### Configuration Hierarchy

1. **`XArmEnvCfg`** (`xarm_env_cfg.py`): base environment — pilot type, observation order, domain randomization (`DomainRandCfg`), control thresholds (`CtrlCfg`), 128 parallel envs
2. **Task cfg** (`assembly_tasks_cfg.py`): task-specific assets, reward scales, success thresholds, data paths
3. **RL-Games PPO** (`agents/rl_games_ppo_cfg.yaml`): γ=0.99, τ=0.95, lr=5e-4 (adaptive), 32-step horizon, 2-layer MLP [32, 32]

### Pilot Models

- **KNN** (`pilot_models/knn_pilot.py`): finds K=10 nearest neighbors in training data, interpolates actions with variable-horizon queuing (1–15 steps). Config: `config/knn_cfg.json`.
- **BC/DiffusionPolicy** (`pilot_models/bc_pilot.py`): wraps LeRobot's DiffusionPolicy. Two variants: `bc_teleop` (teleop data) and `bc_expert` (expert data). Config: `config/state_dp_cfg.json` or `vision_dp_cfg.json`.

Set via `XArmEnvCfg.pilot_model` ∈ `{"knn", "bc_teleop", "bc_expert"}` and `XArmEnvCfg.pilot_type` ∈ `{"noisy", "laggy"}`.

### Key Files

| File | Role |
|------|------|
| `source/xarm_assembly_env/xarm_env.py` | Main `DirectRLEnv` — scene setup, step loop, rewards, reset |
| `source/xarm_assembly_env/xarm_env_cfg.py` | All config dataclasses (`XArmEnvCfg`, `DomainRandCfg`, `CtrlCfg`, obs/state dim maps) |
| `source/xarm_assembly_env/assembly_tasks_cfg.py` | Task-specific asset + reward configs |
| `source/utils/control.py` | `adm_ctrl_task_space()` (admittance law) + `IK_Controller` (Pinocchio) |
| `source/utils/utils.py` | `resolve_hf()` (HuggingFace asset downloads), math helpers |
| `source/pilot_models/knn_pilot.py` | KNN pilot |
| `source/pilot_models/bc_pilot.py` | DiffusionPolicy BC pilot |
| `scripts/train.py` | RL training entry point |

### Data & Assets

All training data, model checkpoints, and USD/URDF assets are fetched from HuggingFace repo `shashuo0104/residual_copilot_data` via `resolve_hf()` in `utils.py`. Paths are specified in task configs.

### Domain Randomization

Admittance control parameters (`Kx`, `Kr`, `mx`, `mr`) are randomized per episode when `dmr.use_dmr=True`. This is the primary source of sim-to-real robustness.
