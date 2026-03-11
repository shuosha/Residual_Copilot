#!/bin/bash
# Train a DiffusionPolicy (BC) pilot model using LeRobot.
#
# Usage:
#   bash scripts/train_bc.sh <repo_id> <job_name>
#
# Arguments:
#   repo_id   — HuggingFace dataset repo ID (see below)
#   job_name  — arbitrary tag appended to the output dir and wandb run name
#
# Available datasets (HF collection: shashuo0104/residual-copilot):
#
#   Expert demonstrations:
#     shashuo0104/0129_peginsert_expert_2000
#     shashuo0104/0129_nutthread_expert_2000
#     shashuo0104/0126_gearmesh_expert_2000
#
#   Augmented teleoperation:
#     shashuo0104/0121_peginsert_teleop_aug_2000
#     shashuo0104/0121_gearmesh_teleop_aug_2000
#     shashuo0104/0121_nutthread_teleop_aug_2000
#
# Examples:
#   bash scripts/train_bc.sh shashuo0104/0129_peginsert_expert_2000 peg_expert
#   bash scripts/train_bc.sh shashuo0104/0121_gearmesh_teleop_aug_2000 gear_teleop

# NOTE: Preload system libgomp to avoid "cannot allocate memory in static TLS
# block" from sklearn's bundled copy when running under Isaac Sim.
export LD_PRELOAD="/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+ $LD_PRELOAD}"

repo_id=$1
job_name=$2

python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
    --config_path source/pilot_models/config/state_dp_cfg.json \
    --dataset.repo_id $repo_id \
    --output_dir outputs/train/${repo_id}_${job_name} \
    --job_name ${repo_id}_${job_name} \