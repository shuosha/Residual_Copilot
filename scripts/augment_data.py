#!/usr/bin/env python3
"""Augment a .npy episode dataset with random SE(2) perturbations (xy translation + yaw).

Input/output format:
  { 'episode_0000': { 'obs.fingertip_pos': (T,3), 'obs.fingertip_quat': (T,4), ... }, ... }

Copies all originals first, then samples augmented episodes (from originals only)
until reaching --target-total. Each augmented episode gets a constant random
(dx, dy) ~ N(0, pos_aug²) and yaw ~ N(0, rot_aug_rad²) applied to pos and quat fields.

Example:
  python augment_data.py --in data.npy --out data_aug.npy \
    --target-total 2000 --pos-aug 0.01 --rot-aug-deg 5 --quat-order wxyz --seed 0
"""
import argparse
import copy
import math
from typing import Dict

import numpy as np

# Quaternion component indices for each convention
_QUAT_IDX = {
    "xyzw": {"x": 0, "y": 1, "z": 2, "w": 3},
    "wxyz": {"w": 0, "x": 1, "y": 2, "z": 3},
}


def quat_mul(q: np.ndarray, r: np.ndarray, order: str) -> np.ndarray:
    """Hamilton product q * r for either 'xyzw' or 'wxyz' convention."""
    idx = _QUAT_IDX[order]
    w1, x1, y1, z1 = q[..., idx["w"]:idx["w"]+1], q[..., idx["x"]:idx["x"]+1], q[..., idx["y"]:idx["y"]+1], q[..., idx["z"]:idx["z"]+1]
    w2, x2, y2, z2 = r[..., idx["w"]:idx["w"]+1], r[..., idx["x"]:idx["x"]+1], r[..., idx["y"]:idx["y"]+1], r[..., idx["z"]:idx["z"]+1]
    out = np.empty_like(q)
    out[..., idx["w"]:idx["w"]+1] = w1*w2 - x1*x2 - y1*y2 - z1*z2
    out[..., idx["x"]:idx["x"]+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
    out[..., idx["y"]:idx["y"]+1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
    out[..., idx["z"]:idx["z"]+1] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return out


def quat_from_yaw(yaw: float, T: int, order: str) -> np.ndarray:
    """(T, 4) pure-yaw quaternion."""
    idx = _QUAT_IDX[order]
    q = np.zeros((T, 4), dtype=np.float32)
    q[:, idx["w"]] = math.cos(0.5 * yaw)
    q[:, idx["z"]] = math.sin(0.5 * yaw)
    return q


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def augment_episode(
    ep: Dict[str, np.ndarray],
    pos_aug: float,
    rot_aug_rad: float,
    rng: np.random.Generator,
    quat_order: str,
) -> Dict[str, np.ndarray]:
    ep2 = copy.deepcopy(ep)
    trans = rng.normal(0.0, pos_aug, size=2).astype(np.float32)
    yaw = float(rng.normal(0.0, rot_aug_rad))

    for k in ("obs.fingertip_pos", "action.fingertip_pos"):
        if k in ep2:
            ep2[k] = ep2[k].copy().astype(np.float32)
            ep2[k][:, :2] += trans

    for k in ("obs.fingertip_quat", "action.fingertip_quat"):
        if k in ep2:
            q = ep2[k].copy().astype(np.float32)
            ep2[k] = normalize_quat(quat_mul(q, quat_from_yaw(yaw, q.shape[0], quat_order), quat_order))

    return ep2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="Input data.npy")
    ap.add_argument("--target-total", type=int, required=True, help="Total episodes in output (includes originals)")
    ap.add_argument("--out", dest="out_path", default=None, help="Output path (default: <input>_aug.npy)")
    ap.add_argument("--pos-aug", type=float, default=0.02, help="Std dev for xy translation noise")
    ap.add_argument("--rot-aug-deg", type=float, default=2.0, help="Std dev for yaw noise in degrees")
    ap.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="wxyz", help="Quaternion storage order")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = args.out_path or args.in_path.replace(".npy", "_aug.npy")

    data_in = np.load(args.in_path, allow_pickle=True).item()
    if not isinstance(data_in, dict) or len(data_in) == 0:
        raise ValueError("Input .npy must contain a non-empty dict of episodes")

    orig_keys = sorted(data_in.keys())
    N_orig = len(orig_keys)
    if args.target_total < N_orig:
        raise ValueError(f"target_total ({args.target_total}) < original episodes ({N_orig})")

    num_aug = args.target_total - N_orig
    width = max(4, len(str(args.target_total - 1)))
    rng = np.random.default_rng(args.seed)
    rot_aug_rad = math.radians(args.rot_aug_deg)

    data_out = {}
    for i, k in enumerate(orig_keys):
        data_out[f"episode_{i:0{width}d}"] = data_in[k]
    for j in range(num_aug):
        data_out[f"episode_{N_orig + j:0{width}d}"] = augment_episode(
            data_in[rng.choice(orig_keys)], args.pos_aug, rot_aug_rad, rng, args.quat_order,
        )

    np.save(out_path, data_out, allow_pickle=True)
    print(f"Saved {len(data_out)} episodes ({N_orig} original + {num_aug} augmented) → {out_path}")


if __name__ == "__main__":
    main()
