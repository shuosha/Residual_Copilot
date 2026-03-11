#!/usr/bin/env python3
"""Overlay data points (fingertip positions) onto a camera image with kNN density coloring."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from camera_params import CAM2BASE, K


def load_points_from_json(data_root: Path) -> np.ndarray:
    """Load fingertip positions from episode_*/robot/*.json → obs[:3]."""
    pts = []
    for jf in sorted(data_root.glob("episode_*/robot/*.json")):
        with open(jf) as f:
            pts.append(np.asarray(json.load(f)["obs"][:3], dtype=np.float32))
    if not pts:
        raise FileNotFoundError(f"No files under {data_root}/episode_*/robot/*.json")
    return np.stack(pts)


def load_points_from_npy(npy_path: Path, key="obs.fingertip_pos") -> np.ndarray:
    """Load points from data.npy: {episode: {key: (T,3), ...}}."""
    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("Expected dict at top level of .npy")
    pts = [np.asarray(ep[key], dtype=np.float32)[:, :3]
           for ep in data.values() if key in ep]
    if not pts:
        raise ValueError(f"No '{key}' found in npy file")
    return np.concatenate(pts)


def project_to_image(points_base: np.ndarray) -> np.ndarray:
    """Project base-frame 3D points → image pixels (u, v)."""
    base2cam = np.linalg.inv(CAM2BASE)
    Pc = (base2cam[:3, :3] @ points_base.T + base2cam[:3, 3:4]).T
    mask = Pc[:, 2] > 1e-6
    Pc = Pc[mask]
    u = K[0, 0] * (Pc[:, 0] / Pc[:, 2]) + K[0, 2]
    v = K[1, 1] * (Pc[:, 1] / Pc[:, 2]) + K[1, 2]
    return np.column_stack([u, v])


def knn_density(uv: np.ndarray, k: int = 30) -> np.ndarray:
    """kNN log-density in pixel space, normalized to [0, 1]."""
    if uv.shape[0] < k + 1:
        return np.ones(uv.shape[0], dtype=np.float32)
    dists, _ = NearestNeighbors(n_neighbors=k).fit(uv).kneighbors(uv)
    dens = np.log(1.0 / (dists[:, -1] ** 2 + 1e-12) + 1e-12)
    lo, hi = np.percentile(dens, [5, 95])
    return np.clip((dens - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="RGB image from the camera")
    ap.add_argument("--data-root", type=str, help="Root with episode_*/robot/*.json")
    ap.add_argument("--npy-path", type=str, help="Path to data.npy (overrides --data-root)")
    ap.add_argument("--key", type=str, default="obs.fingertip_pos")
    ap.add_argument("--out", type=str, default="logs/vis/overlay.png")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-points", type=int, default=100_000)
    args = ap.parse_args()

    if args.npy_path is not None:
        pts = load_points_from_npy(Path(args.npy_path), key=args.key)
    elif args.data_root is not None:
        pts = load_points_from_json(Path(args.data_root))
    else:
        raise ValueError("Must provide either --data-root or --npy-path")

    if pts.shape[0] > args.max_points:
        pts = pts[np.random.choice(pts.shape[0], args.max_points, replace=False)]

    img = np.asarray(Image.open(args.image).convert("RGB"))
    H, W = img.shape[:2]

    uv = project_to_image(pts)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[inb]

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter(uv[:, 0], uv[:, 1], c=knn_density(uv, k=args.k),
                cmap="viridis", s=1, alpha=0.2, linewidths=0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
