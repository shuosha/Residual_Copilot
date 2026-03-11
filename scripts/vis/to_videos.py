"""Create single-episode and collage videos from a recorded rollout directory.

Optionally annotates each frame with action-triangle overlays (same as
vis_qual_results.py draws online) using the --annotate flag.

Recording directory layout (produced by scripts/play.py):
    recording_dir/
        meta/infos.json          # {"task": "PegInsert", ...}
        meta/stats.json          # {"episode_0000": {"success": true, ...}, ...}
        episode_0000/
            camera_0/rgb/000000.jpg ...
            robot/000000.json ...

Usage:
    # Single + collage videos, no annotation
    python scripts/visualizations/to_videos.py logs/rollouts/eval_PegInsert_with_...

    # Only single videos, with annotation overlays
    python scripts/visualizations/to_videos.py logs/rollouts/eval_PegInsert_with_... \\
        --single --annotate

    # Only collage
    python scripts/visualizations/to_videos.py logs/rollouts/eval_PegInsert_with_... \\
        --collage --cols 5 --scale 0.25
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from camera_params import INTR, EXTR


# ---------------------------------------------------------------------------
# Task inference from recording directory
# ---------------------------------------------------------------------------
_TASK_CANONICAL = {
    "PegInsert": "peg_insert",
    "GearMesh": "gear_mesh",
    "NutThread": "nut_thread",
    "peg_insert": "peg_insert",
    "gear_mesh": "gear_mesh",
    "nut_thread": "nut_thread",
}


def infer_task_name(root: Path) -> str:
    """Infer canonical task name from meta/infos.json or directory name."""
    infos_path = root / "meta" / "infos.json"
    if infos_path.is_file():
        with open(infos_path) as f:
            infos = json.load(f)
        raw = infos.get("task", "")
        if raw in _TASK_CANONICAL:
            return _TASK_CANONICAL[raw]

    # Fallback: parse eval_{Task}_... from directory name
    m = re.search(r"eval_(PegInsert|GearMesh|NutThread)", root.name)
    if m:
        return _TASK_CANONICAL[m.group(1)]

    raise ValueError(
        f"Cannot infer task from {root}. "
        "Ensure meta/infos.json exists or directory name contains the task."
    )


# ---------------------------------------------------------------------------
# Projection helpers (same as vis_qual_results.py)
# ---------------------------------------------------------------------------
def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_h = np.concatenate([p, np.ones((p.shape[0], 1), dtype=p.dtype)], axis=1)
    return (T @ p_h.T).T[:, :3]


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], np.clip(points_cam[:, 2], 1e-6, None)
    u = K[0, 0] * X / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return np.stack([u, v], axis=1)


def project_base_points_to_uv(points_base, intr_mat, extr_mat):
    T_base_cam = np.linalg.inv(extr_mat)
    pts_cam = transform_points(T_base_cam, points_base)
    valid = pts_cam[:, 2] > 1e-6
    uvs = project_points(pts_cam, intr_mat)
    return uvs, valid


# ---------------------------------------------------------------------------
# Drawing helpers (same as vis_qual_results.py)
# ---------------------------------------------------------------------------
SOFT_RED = (87, 110, 237)
PINK = (193, 131, 163)
SOFT_BLUE = (248, 159, 71)
GRAY = (200, 200, 200)


def draw_line_or_arrow(img, p0, p1, bgr, thickness=2, arrow=True):
    p0 = (int(round(p0[0])), int(round(p0[1])))
    p1 = (int(round(p1[0])), int(round(p1[1])))
    if arrow:
        cv2.arrowedLine(img, p0, p1, bgr, thickness=thickness, tipLength=0.15)
    else:
        cv2.line(img, p0, p1, bgr, thickness=thickness)
    return img


def draw_triangle_overlay(img_bgr, curr_uv, base_uv, net_uv, arrow=True):
    draw_line_or_arrow(img_bgr, curr_uv, base_uv, SOFT_RED, thickness=2, arrow=arrow)
    draw_line_or_arrow(img_bgr, base_uv, net_uv, PINK, thickness=2, arrow=arrow)
    draw_line_or_arrow(img_bgr, curr_uv, net_uv, SOFT_BLUE, thickness=2, arrow=arrow)
    return img_bgr


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw_np(q: np.ndarray, order: str = "wxyz") -> float:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return wrap_to_pi(math.atan2(siny_cosp, cosy_cosp))


def draw_yaw_overlay_3d(
    img_bgr, curr_pos_base, base_yaw, env_yaw,
    intr_mat, extr_mat, r_m=0.03, thickness=2, alpha=0.6,
):
    center = np.asarray(curr_pos_base, dtype=np.float64).reshape(3)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def pt_on_circle(theta):
        return center + r_m * (math.cos(theta) * ex + math.sin(theta) * ey)

    n_circle = 80
    thetas = np.linspace(0.0, 2.0 * math.pi, n_circle, endpoint=True)
    circle_pts_base = np.stack([pt_on_circle(t) for t in thetas], axis=0)

    delta = wrap_to_pi(env_yaw - base_yaw)
    n_arc = 40
    arc_thetas = np.linspace(base_yaw, base_yaw + delta, n_arc, endpoint=True)
    arc_pts_base = np.stack([pt_on_circle(t) for t in arc_thetas], axis=0)

    all_pts_base = np.concatenate([center[None, :], circle_pts_base, arc_pts_base], axis=0)
    uvs, valid = project_base_points_to_uv(all_pts_base, intr_mat, extr_mat)
    if not bool(valid.all()):
        return img_bgr

    idx = 1  # skip center_uv
    circle_uvs = uvs[idx:idx + n_circle]; idx += n_circle
    arc_uvs = uvs[idx:idx + n_arc]

    overlay = np.zeros_like(img_bgr)
    circle_poly = np.round(circle_uvs).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [circle_poly], isClosed=False, color=GRAY, thickness=1)

    arc_poly = np.round(arc_uvs).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [arc_poly], isClosed=False, color=PINK, thickness=thickness)

    cv2.circle(overlay, tuple(arc_poly[0, 0].tolist()), 3, PINK, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, tuple(arc_poly[-1, 0].tolist()), 3, PINK, -1, lineType=cv2.LINE_AA)

    mask = overlay.sum(axis=2) > 0
    out = img_bgr.copy()
    out[mask] = ((1.0 - alpha) * img_bgr[mask] + alpha * overlay[mask]).astype(np.uint8)
    return out


def draw_task_overlay(img_bgr, robot_data: dict, task_name: str, intr_mat, extr_mat):
    curr_pos = np.array(robot_data["obs.fingertip_pos"], dtype=np.float64)
    curr_q = np.array(robot_data["obs.fingertip_quat"], dtype=np.float64)
    base_pos = np.array(robot_data["base_action.fingertip_pos"], dtype=np.float64)
    base_q = np.array(robot_data["base_action.fingertip_quat"], dtype=np.float64)
    has_env_action = "action.fingertip_pos" in robot_data

    if has_env_action:
        if task_name == "nut_thread":
            base_yaw = quat_to_yaw_np(base_q, order="wxyz")
            env_yaw = quat_to_yaw_np(curr_q, order="wxyz")
            img_bgr = draw_yaw_overlay_3d(
                img_bgr, curr_pos_base=curr_pos,
                base_yaw=base_yaw, env_yaw=env_yaw,
                intr_mat=intr_mat, extr_mat=extr_mat,
            )
        net_pos = np.array(robot_data["action.fingertip_pos"], dtype=np.float64)
        pts_base = np.stack([curr_pos, base_pos, net_pos], axis=0)
        uvs, valid = project_base_points_to_uv(pts_base, intr_mat, extr_mat)
        if bool(valid.all()):
            return draw_triangle_overlay(img_bgr, uvs[0], uvs[1], uvs[2], arrow=True)
    else:
        pts_base = np.stack([curr_pos, base_pos], axis=0)
        uvs, valid = project_base_points_to_uv(pts_base, intr_mat, extr_mat)
        if bool(valid.all()):
            return draw_line_or_arrow(img_bgr, uvs[0], uvs[1], SOFT_RED, thickness=2, arrow=True)

    return img_bgr


# ---------------------------------------------------------------------------
# Video helpers (adapted from dir_to_video.py)
# ---------------------------------------------------------------------------
def _videos_dir(root: Path) -> Path:
    d = root / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _natsorted_jpgs(rgb_dir: Path, pattern: str = "*.jpg") -> List[Path]:
    paths = list(rgb_dir.glob(pattern))
    def key(p: Path):
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else p.name
    return sorted(paths, key=key)


def _list_episodes(root: Path) -> List[Path]:
    eps = [p for p in root.glob("episode_*") if p.is_dir()]
    def key(p: Path):
        m = re.findall(r"\d+", p.name)
        return int(m[0]) if m else 10**9
    return sorted(eps, key=key)


def _list_cameras(ep_dir: Path) -> List[int]:
    cams = []
    for d in ep_dir.glob("camera_*"):
        if (d / "rgb").is_dir():
            try:
                cams.append(int(d.name.split("_", 1)[1]))
            except Exception:
                pass
    return sorted(cams)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _crop(img: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return img
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]


def _make_even_hw(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h % 2 == 1:
        img = img[:-1, :, :]
    if w % 2 == 1:
        img = img[:, :-1, :]
    return img


def _load_stats(root: Path) -> Optional[Dict[str, dict]]:
    """Load meta/stats.json → {episode_name: {"success": bool, ...}}."""
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        return None
    with open(stats_path) as f:
        return json.load(f)


def _load_robot_data_for_episode(ep_dir: Path) -> Dict[str, dict]:
    """Load all robot/*.json for an episode, keyed by zero-padded stem."""
    robot_dir = ep_dir / "robot"
    out = {}
    if not robot_dir.is_dir():
        return out
    for p in sorted(robot_dir.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def _read_and_annotate(
    jpg_path: Path,
    robot_data: Optional[dict],
    task_name: str,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Read a JPG, optionally crop and annotate, return BGR image (or None)."""
    img = cv2.imread(str(jpg_path))
    if img is None:
        return None
    if robot_data is not None:
        img = draw_task_overlay(img, robot_data, task_name, INTR, EXTR)
    img = _crop(img, crop)
    return img


# ---------------------------------------------------------------------------
# Single-episode videos
# ---------------------------------------------------------------------------
def make_single_videos(
    root: Path,
    fps: int = 15,
    cam_idx: Optional[int] = None,
    annotate: bool = False,
    task_name: str = "",
    crop: Optional[Tuple[int, int, int, int]] = None,
    episode_filter: Optional[List[int]] = None,
) -> List[Path]:
    """Create H.264 per-episode videos under videos/.

    If meta/stats.json exists, prefix filenames with success_ / failure_.
    When *annotate* is True, draw action overlays on each frame.
    """
    videos = _videos_dir(root)
    outputs: List[Path] = []
    episodes = _list_episodes(root)
    if episode_filter is not None:
        wanted = {f"episode_{i:04d}" for i in episode_filter}
        episodes = [ep for ep in episodes if ep.name in wanted]
    if not episodes:
        print("[warn] no episode_* found")
        return outputs

    stats = _load_stats(root)

    cam_set = set()
    for ep in episodes:
        for c in _list_cameras(ep):
            if cam_idx is None or c == cam_idx:
                cam_set.add(c)
    cams = sorted(cam_set)
    if not cams:
        print(f"[warn] no cameras found (filter cam_idx={cam_idx})")
        return outputs

    for ep in episodes:
        m = re.findall(r"\d+", ep.name)
        ep_num = int(m[0]) if m else 0

        label = None
        if stats is not None and ep.name in stats:
            label = "success" if stats[ep.name].get("success", False) else "failure"

        robot_cache = _load_robot_data_for_episode(ep) if annotate else {}

        for c in cams:
            rgb_dir = ep / f"camera_{c}" / "rgb"
            if not rgb_dir.is_dir():
                continue

            suffix = "_annotated" if annotate else ""
            if label is not None:
                out_mp4 = videos / f"{label}_eps_{ep_num:04d}_cam_{c}{suffix}.mp4"
            else:
                out_mp4 = videos / f"eps_{ep_num:04d}_cam_{c}{suffix}.mp4"

            jpgs = _natsorted_jpgs(rgb_dir)
            if not jpgs:
                continue

            # Determine frame size from first frame
            first = _read_and_annotate(
                jpgs[0],
                robot_cache.get(jpgs[0].stem) if annotate else None,
                task_name, crop=crop,
            )
            if first is None:
                continue
            first = _make_even_hw(first)
            H, W = first.shape[:2]

            last_stem = jpgs[-1].stem if jpgs else None

            def frames(jpgs=jpgs, robot_cache=robot_cache, last_stem=last_stem, label=label) -> Iterable[np.ndarray]:
                for p in jpgs:
                    rd = robot_cache.get(p.stem) if (annotate and p.stem != last_stem) else None
                    img = _read_and_annotate(p, rd, task_name, crop=crop)
                    if img is None:
                        continue
                    img = _make_even_hw(img)
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                        img = _make_even_hw(img)
                    if annotate and p.stem == last_stem and label is not None:
                        border_bgr = (0, 200, 0) if label == "success" else (0, 0, 200)
                        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                                      border_bgr, thickness=4)
                    yield _bgr_to_rgb(img)

            with imageio.get_writer(
                str(out_mp4), fps=fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=None,
            ) as w:
                for f in frames():
                    w.append_data(f)

            print(f"[ok] {out_mp4}")
            outputs.append(out_mp4)

    return outputs


# ---------------------------------------------------------------------------
# Collage videos
# ---------------------------------------------------------------------------
def make_collage_videos(
    root: Path,
    fps: int = 15,
    cols: int = 5,
    cam_idx: Optional[int] = None,
    scale: float = 0.25,
    annotate: bool = False,
    task_name: str = "",
    crop: Optional[Tuple[int, int, int, int]] = None,
    episode_filter: Optional[List[int]] = None,
) -> None:
    """Create collage grid videos, one per camera."""
    videos = _videos_dir(root)
    episodes = _list_episodes(root)
    if episode_filter is not None:
        wanted = {f"episode_{i:04d}" for i in episode_filter}
        episodes = [ep for ep in episodes if ep.name in wanted]
    if not episodes:
        print("[warn] no episode_* found")
        return

    stats = _load_stats(root)

    cam_set = set()
    for ep in episodes:
        for c in _list_cameras(ep):
            if cam_idx is None or c == cam_idx:
                cam_set.add(c)
    cams = sorted(cam_set)
    if not cams:
        print("[warn] no cameras found for collage")
        return

    # Pre-load robot data if annotating
    ep_robot_caches = {}
    if annotate:
        for ep in episodes:
            ep_robot_caches[ep.name] = _load_robot_data_for_episode(ep)

    for c in cams:
        suffix = "_annotated" if annotate else ""
        out_mp4 = videos / f"collage_cam_{c}{suffix}.mp4"

        # Gather per-episode JPG lists for this camera
        ep_jpgs: List[Tuple[Path, List[Path]]] = []  # (ep_dir, jpg_list)
        for ep in episodes:
            rgb_dir = ep / f"camera_{c}" / "rgb"
            if rgb_dir.is_dir():
                lst = _natsorted_jpgs(rgb_dir)
                if lst:
                    ep_jpgs.append((ep, lst))

        if not ep_jpgs:
            continue

        # Determine cell size from first readable frame
        first_frame = None
        for ep, lst in ep_jpgs:
            rd = None
            if annotate:
                rd = ep_robot_caches.get(ep.name, {}).get(lst[0].stem)
            f0 = _read_and_annotate(lst[0], rd, task_name, crop=crop)
            if f0 is None:
                continue
            if scale != 1.0:
                f0 = cv2.resize(f0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            f0 = _make_even_hw(f0)
            first_frame = f0
            break

        if first_frame is None:
            print(f"[skip] camera {c}: no readable frames")
            continue

        H, W = first_frame.shape[:2]
        rows = math.ceil(len(ep_jpgs) / cols)
        canvas_h = rows * H - (rows * H) % 2
        canvas_w = cols * W - (cols * W) % 2

        lens = [len(lst) for _, lst in ep_jpgs]
        max_len = max(lens)

        def collage_frames(
            ep_jpgs=ep_jpgs, lens=lens, ep_robot_caches=ep_robot_caches,
            stats=stats,
        ) -> Iterable[np.ndarray]:
            for t in range(max_len):
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                for k, (ep, lst) in enumerate(ep_jpgs):
                    r, col_ = divmod(k, cols)
                    y0, y1 = r * H, (r + 1) * H
                    x0, x1 = col_ * W, (col_ + 1) * W
                    if y1 > canvas_h or x1 > canvas_w:
                        continue

                    idx = min(t, lens[k] - 1)
                    p = lst[idx]
                    rd = None
                    if annotate and idx < lens[k] - 1:
                        rd = ep_robot_caches.get(ep.name, {}).get(p.stem)
                    img = _read_and_annotate(p, rd, task_name, crop=crop)
                    if img is None:
                        continue
                    if scale != 1.0:
                        img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                    img = _make_even_hw(img)
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                        img = _make_even_hw(img)
                    if annotate and idx == lens[k] - 1 and stats is not None:
                        is_success = stats.get(ep.name, {}).get("success", False)
                        border_bgr = (0, 200, 0) if is_success else (0, 0, 200)
                        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                                      border_bgr, thickness=4)
                    canvas[y0:y1, x0:x1] = _bgr_to_rgb(img)
                yield canvas

        with imageio.get_writer(
            str(out_mp4), fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None,
        ) as w:
            for f in collage_frames():
                w.append_data(f)

        print(f"[ok] {out_mp4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Create per-episode and collage videos from a recorded rollout directory.",
    )
    ap.add_argument("root", help="Path to recording_dir/ containing episode_XXXX/")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cam-idx", type=int, default=None, help="Only process this camera index")
    ap.add_argument("--cols", type=int, default=5, help="Collage grid columns")
    ap.add_argument("--scale", type=float, default=0.25, help="Collage cell downscale factor")
    ap.add_argument("--single", action="store_true", help="Only make single-episode videos")
    ap.add_argument("--collage", action="store_true", help="Only make collage videos")
    ap.add_argument("--annotate", action="store_true",
                    help="Draw action-triangle overlays on each frame")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    metavar="N", help="Only process these episode indices (e.g. --episodes 0 3 7)")
    ap.add_argument("--crop", type=int, nargs=4,
                    default=[50, -180, 230, -300],
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="Crop each frame: y0 y1 x0 x1 in pixel coords (applied after annotation)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] {root} does not exist")
        return

    # Default: both single and collage
    if not args.single and not args.collage:
        args.single = True
        args.collage = True

    crop = tuple(args.crop) if args.crop is not None else None

    task_name = ""
    if args.annotate:
        task_name = infer_task_name(root)

    if args.single:
        make_single_videos(
            root, fps=args.fps, cam_idx=args.cam_idx,
            annotate=args.annotate, task_name=task_name, crop=crop,
            episode_filter=args.episodes,
        )

    if args.collage:
        make_collage_videos(
            root, fps=args.fps, cols=args.cols, cam_idx=args.cam_idx,
            scale=args.scale, annotate=args.annotate, task_name=task_name,
            crop=crop, episode_filter=args.episodes,
        )


if __name__ == "__main__":
    main()
