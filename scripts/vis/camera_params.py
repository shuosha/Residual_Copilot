"""Canonical camera intrinsics and extrinsics (matching CameraCfg in xarm_env_cfg.py)."""

import numpy as np
from scipy.spatial.transform import Rotation

INTR = np.array([
    [426.7812194824219, 0.0, 425.43218994140625],
    [0.0, 426.23809814453125, 245.81968688964844],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

_Q_WXYZ = np.array([-0.3464, 0.6371, 0.6027, -0.3330], dtype=np.float64)
_T_XYZ = np.array([0.7263, -0.0323, 0.2216], dtype=np.float64)


def _build_extr_cam2base(q_wxyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 cam-to-base transform from wxyz quaternion + translation."""
    w, x, y, z = q_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy uses xyzw
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


EXTR = _build_extr_cam2base(_Q_WXYZ, _T_XYZ)

# Aliases used by plot_data.py
CAM2BASE = EXTR
K = INTR
