"""Simple wrapper around a DiffusionPolicy and its pre/post processors.

This provides a tiny inference helper used for quick local testing. It:
 - loads a pretrained DiffusionPolicy from `model_id` (a local folder or HF repo)
 - loads/creates pre- and post-processors (optionally using dataset stats from `dataset_id`)
 - exposes `act(obs)` which runs preprocessor -> model.select_action -> postprocessor

The wrapper deliberately keeps the surface area small. It assumes the input `obs` is already in
LeRobot observation format (keys that the pipeline expects, typically starting with
`observation.*`). The returned value is whatever the postprocessor returns (usually a tensor or
numpy array representing the robot action).
"""
from __future__ import annotations

import os, json
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_action_normalizer
from lerobot.policies.factory import make_pre_post_processors

class BC_Pilot:
    """Minimal DiffusionPolicy + processors wrapper.

    Args:
        model_id: pretrained model folder or HF repo id containing the policy (config + weights).
        dataset_id: optional dataset repo id (used to load dataset stats for processors).
        device: torch device string (e.g. 'cpu' or 'cuda'). If None, the policy config's device is used.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        # Load training config to get policy horizon
        training_cfg = os.path.join(model_id, "train_config.json")
        with open(training_cfg, "r") as f:
            train_config = json.load(f)

        # Load the pretrained diffusion policy
        self.policy: DiffusionPolicy = DiffusionPolicy.from_pretrained(model_id)

        # Prefer explicit device argument if provided
        self.device = "cuda" if device is None else device

        self.policy.to(self.device)
        self.policy.eval()

        # Create / load pre and post processors. Normalization stats are loaded from the checkpoint.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id,
        )

        # Load the reference action normalizer from checkpoint (stats are already embedded)
        self.ref_action_processor = make_action_normalizer(
            config=self.policy.config,
            pretrained_path=model_id,
        )

    def act(self, obs: Dict[str, Any], ref_action: Tensor | None = None):
        """Run a single observation through preprocessor, model and postprocessor.

        The `obs` argument should be in the LeRobot observation format expected by the processors
        (for example, keys like `observation.state`, `observation.images`, ...). The method returns
        the postprocessed action (usually a tensor or numpy array).
        """
        # Preprocess observation (converts to batched tensors and moves to device)
        processed = self.preprocessor(obs)

        if ref_action is not None:
            ref_action = self.ref_action_processor(ref_action)

        # Run model inference
        with torch.inference_mode():
            action_tensor = self.policy.select_action(processed, ref_action=ref_action)

        # Postprocess (unnormalize, move to CPU / numpy as configured by the pipeline)
        postprocessed = self.postprocessor(action_tensor).to(self.device)

        return postprocessed
    
    def reset(self):
        """Reset any internal stateful components (e.g. normalizers)."""
        self.policy.reset()
        self.idx = 0


__all__ = ["BC_Pilot"]