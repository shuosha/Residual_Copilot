import sapien.core as sapien
import torch
import numpy as np
import isaacsim.core.utils.torch as torch_utils # type: ignore
from isaaclab.utils.math import axis_angle_from_quat, quat_from_angle_axis

def get_task_space_error(
    cur_pos, cur_quat,
    tgt_pos, tgt_quat,
    jacobian_type="geometric",
    rot_error_type="axis_angle",
):
    """
    Returns errors as (current - target) for BOTH position and orientation.
    Assumes unit quaternions. Uses shortest-arc convention.
    """
    # 1) Position error: current - target  (matches admittance F_sd = K*e + D*xdot_ref)
    pos_error = cur_pos - tgt_pos

    # 2) Orientation error: current - target
    # Shortest path: flip target if dot<0
    dot = (tgt_quat * cur_quat).sum(dim=1, keepdim=True)
    tgt = torch.where(dot >= 0, tgt_quat, -tgt_quat)

    # For unit quats, inverse == conjugate
    tgt_inv = torch_utils.quat_conjugate(tgt)
    # q_rel = cur ∘ tgt^{-1}  (gives “current − target”)
    q_rel  = torch_utils.quat_mul(cur_quat, tgt_inv)

    if rot_error_type == "quat":
        rot_error = q_rel  # expresses current relative to target
    else:
        # axis-angle from q_rel (already “current − target”)
        rot_error = axis_angle_from_quat(q_rel)

    return pos_error, rot_error


def adm_ctrl_task_space(
    pos, quat,  
    pos_g, quat_g, 
    v, F_ext, dt,
    kx, kr, mx, mr, dx, dr,
):
    B, _ = pos.shape
    device = pos.device

    # task-space error
    pos_err, aa_err = get_task_space_error(
        pos, quat,
        pos_g, quat_g,
        jacobian_type="geometric", rot_error_type="axis_angle",
    )
    e = torch.cat((pos_err, aa_err), dim=1)  # (B,6)

    # per-env scalars (B,)
    def to_B(x):
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        return x.expand(B) if x.ndim == 0 else x

    Kx = to_B(kx);  Kr = to_B(kr)
    Mx = to_B(mx);  Mr = to_B(mr)
    Dx = to_B(dx);  Dr = to_B(dr)

    # build 6D vectors from (B,)
    K = torch.stack([Kx, Kx, Kx, Kr, Kr, Kr], dim=1)    # (B,6)
    D = torch.stack([Dx, Dx, Dx, Dr, Dr, Dr], dim=1)    # (B,6)
    M = torch.stack([Mx, Mx, Mx, Mr, Mr, Mr], dim=1)    # (B,6)

    # admittance update
    F_sd  = K * e + D * v
    vdot = (F_ext - F_sd) / M
    v = v + dt * vdot

    pos_cmd = pos + dt * v[:, :3]
    angle = (v[:, 3:] * dt).norm(dim=1)                    # (N,)
    axis  = (v[:, 3:] * dt) / (angle.unsqueeze(1) + 1e-8)  # (N, 3)
    dq = quat_from_angle_axis(angle, axis)
    quat_cmd = torch_utils.quat_mul(dq, quat)  # quaternion multiplication
    quat_cmd = quat_cmd / quat_cmd.norm(dim=1, keepdim=True)  # normalize

    return torch.cat((pos_cmd, quat_cmd), dim=1), v

class IK_Controller:
    def __init__(self, urdf_path):
        # load sapien robot
        engine = sapien.Engine()
        scene = engine.create_scene()
        loader = scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == "link7":
                self.sapien_eef_idx = link_idx
                break

    def _compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def _compute_ik_sapien(self, initial_qpos, tf, verbose=False):
        """
        Compute IK using sapien
        initial_qpos: (7,) numpy array
        tf: (4, 4) transformation matrix
        """
        pose = sapien.Pose.from_transformation_matrix(tf)

        active_qmask = np.array([True, True, True, True, True, True, True])
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx, 
            pose=pose,
            initial_qpos=initial_qpos, 
            active_qmask=active_qmask, 
            )
        if verbose:
            print('ik qpos:', qpos)

        # verify ik
        fk_pose = self._compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        
        if verbose:
            print('target pose for IK:', tf)
            print('fk pose for IK:', fk_pose)
        
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf[:3, :3])
        
        if pose_diff > 0.01 or rot_diff > 0.01:
            print(f'[IK WARNING] pose_diff={pose_diff:.4f}, rot_diff={rot_diff:.4f} — returning initial_qpos')
            return initial_qpos
        return qpos[0]

    def compute_ik(self, init_qpos, cartesian_target):
        num_envs = init_qpos.shape[0]
        target_qpos = torch.zeros_like(init_qpos)
        for i in range(num_envs):
            curr_qpos = init_qpos[i, 0:7].cpu().numpy()
            tf = np.eye(4)
            tf[:3, :3] = torch_utils.quats_to_rot_matrices(cartesian_target[i, 3:7]).cpu().numpy()  
            tf[:3, 3] = cartesian_target[i, :3].cpu().numpy()
            ik_qpos = self._compute_ik_sapien(
                initial_qpos=curr_qpos,
                tf=tf,
                verbose=False,
            )
            target_qpos[i, 0:7] = torch.tensor(ik_qpos, device=init_qpos.device)
        return target_qpos