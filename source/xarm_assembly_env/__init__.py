# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

from .xarm_env_cfg import (
    XArmGearMeshCfg, XArmNutThreadCfg, XArmPegInsertCfg,
    XArmGearMeshGuidedDiffusionCfg, XArmNutThreadGuidedDiffusionCfg, XArmPegInsertGuidedDiffusionCfg,
)

"""
id := [robot]-[task]-[method]
"""

gym.register(
    id="XArm-GearMesh-Residual",
    entry_point=f"{__name__}.xarm_env:XArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-NutThread-Residual",
    entry_point=f"{__name__}.xarm_env:XArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-PegInsert-Residual",
    entry_point=f"{__name__}.xarm_env:XArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

##
# GuidedDiffusion environments (8D absolute actions).
##

gym.register(
    id="XArm-GearMesh-GuidedDiffusion",
    entry_point=f"{__name__}.xarm_env_guided_diffusion:XArmEnvGuidedDiffusion",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmGearMeshGuidedDiffusionCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-NutThread-GuidedDiffusion",
    entry_point=f"{__name__}.xarm_env_guided_diffusion:XArmEnvGuidedDiffusion",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmNutThreadGuidedDiffusionCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="XArm-PegInsert-GuidedDiffusion",
    entry_point=f"{__name__}.xarm_env_guided_diffusion:XArmEnvGuidedDiffusion",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmPegInsertGuidedDiffusionCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)