from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError
import isaacsim.core.utils.torch as torch_utils # type: ignore

import torch
import numpy as np

def resolve_hf(
    repo_id: str,
    path: str,
    *,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    materialize: bool = False,
) -> str:
    """
    Resolve a Hugging Face Hub path to a local filesystem path.

    - If `path` is a file: download that file and return its local cached path.
    - If `path` is a directory/prefix: snapshot_download only files under that prefix and return the local dir path.
    - If `materialize=True`: copy to a real temp dir (dereferences symlinks) and return that path.
      Useful for URDF/meshes workflows that dislike HF cache symlinks/hardlinks.

    Returns: str local path (file or directory).
    """
    path = path.strip("/")

    # 1) Try as a file first (fast path).
    try:
        file_local = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
        )
        file_local = Path(file_local)

        if not materialize:
            return str(file_local)

        stage_root = Path(tempfile.mkdtemp(prefix="hf_materialized_"))
        dst = stage_root / file_local.name
        shutil.copy2(file_local, dst)  # dereference into a real file
        return str(dst)

    except EntryNotFoundError:
        # Not a file -> treat as directory/prefix below.
        pass

    # 2) Directory / prefix case.
    prefix = path + "/"
    snap_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[f"{prefix}*"],  # '*' matches across subdirs too
        )
    )

    src_dir = snap_root / prefix
    if not src_dir.is_dir():
        raise FileNotFoundError(f"'{path}' not found as file or directory prefix in {repo_id} ({repo_type})")

    if not materialize:
        return str(src_dir)

    stage_root = Path(tempfile.mkdtemp(prefix="hf_materialized_"))
    dst_dir = stage_root / path
    dst_dir.parent.mkdir(parents=True, exist_ok=True)

    # Copy to a real directory; dereference symlinks by default (symlinks=False).
    shutil.copytree(src_dir, dst_dir, symlinks=False, dirs_exist_ok=True)
    return str(dst_dir)

def build_init_state(data_path: str, num_envs: int, dtype=torch.float32, device="cpu"):
    """
    Returns:
    init: (num_eps, 13) torch tensor
    ep_keys: list[str] episode ordering used
    """
    data = np.load(data_path, allow_pickle=True).item()
    ep_keys = sorted(data.keys())
    num_eps = len(ep_keys)

    init = torch.empty((num_eps, 13), device=device, dtype=dtype)

    for i, k in enumerate(ep_keys):
        ep = data[k]

        pos  = torch.as_tensor(ep["obs.fingertip_pos"][0], device=device, dtype=dtype)   # (3,)
        quat = torch.as_tensor(ep["obs.fingertip_quat"][0], device=device, dtype=dtype)  # (4,)

        rel_fixed = torch.as_tensor(ep["obs.fingertip_pos_rel_fixed"][0], device=device, dtype=dtype)  # (3,)
        rel_held  = torch.as_tensor(ep["obs.fingertip_pos_rel_held"][0],  device=device, dtype=dtype)  # (3,)
        a = pos - rel_fixed  # (3,)
        b = pos - rel_held   # (3,)

        init[i] = torch.cat([pos, quat, a, b], dim=0)  # (13,)

    return init.unsqueeze(0).expand(num_envs, -1, -1)

def collapse_obs_dict(obs_dict, obs_order):
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors


def set_friction(asset, value, num_envs):
    """Update material properties for a given asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)

def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8):
    """
    q1, q2: (..., 4) float tensors in [w, x, y, z] or [x, y, z, w]—either is fine
            as long as both use the same convention.
    Returns: (...,) radians in [0, pi]
    """
    # normalize
    q1 = q1 / (q1.norm(dim=-1, keepdim=True).clamp_min(eps))
    q2 = q2 / (q2.norm(dim=-1, keepdim=True).clamp_min(eps))

    # dot, handle sign ambiguity
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(-1 + eps, 1 - eps)

    return 2.0 * torch.arccos(dot)

def get_held_base_pose(held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get current poses for keypoint and success computation."""
    held_base_pos_local = get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get target poses for keypoint and success computation."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        fixed_success_pos_local[:, 0] = gear_base_offset[0]
        fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif task_name == "nut_thread":
        fixed_success_pos_local[:, 2] = fixed_asset_cfg.nut_offset[2]
    elif task_name == "peg_insert":
        fixed_success_pos_local[:, 2] = 0.0
    else:
        raise NotImplementedError("Task not implemented")
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat

def get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device):
    """Get transform between asset default frame and geometric base frame."""
    held_base_x_offset = 0.0
    if task_name == "peg_insert":
        held_base_z_offset = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif task_name == "nut_thread":
        held_base_z_offset = 0.0
    else:
        raise NotImplementedError("Task not implemented")

    held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset

    return held_base_pos_local