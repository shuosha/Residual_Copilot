import json
import torch, numpy as np
from residual_copilot.utils.utils import quat_geodesic_angle

def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """q0, q1: (..., 4) unit quats; t: (...,) or (..., 1) in [0,1]; returns (..., 4)"""
    if t.ndim == q0.ndim - 1:
        t = t.unsqueeze(-1)

    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(eps)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1 + eps, 1 - eps)

    close = dot > (1.0 - 1e-4)
    q_lin = q0 + t * (q1 - q0)
    q_lin = q_lin / q_lin.norm(dim=-1, keepdim=True).clamp_min(eps)

    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    q_slerp = w0 * q0 + w1 * q1
    q_slerp = q_slerp / q_slerp.norm(dim=-1, keepdim=True).clamp_min(eps)

    return torch.where(close, q_lin, q_slerp)

def _interp_weights(H_max: int, H_env: torch.Tensor, gamma: float, device) -> torch.Tensor:
    """
    Returns w of shape (M, H_max):
      w[:,0] = 0 (first action matches prev chunk last action),
      w reaches 1 at step H_env-1 (end of used horizon is fully new chunk),
      nonlinear bias toward new chunk via u**gamma (gamma<1 biases to new).
    """
    M = H_env.shape[0]
    ar = torch.arange(H_max, device=device).view(1, H_max).expand(M, H_max)
    denom = (H_env - 1).clamp_min(1).view(M, 1)
    w = (ar / denom).clamp(0.0, 1.0).pow(gamma)
    is_one = (H_env == 1).view(M, 1)
    return torch.where(is_one, torch.ones_like(w), w)


class KNN_Pilot:
    """Nearest-neighbor action retriever with per-env horizon queues."""

    def __init__(self, cfg_path: str, data_path: str, num_envs: int, device: str = "cpu", replay_mode: bool = False):
        with open(cfg_path) as f:
            cfg = json.load(f)

        self._device = device
        min_horizon  = cfg["min_horizon"]
        max_horizon  = cfg["max_horizon"]

        if min_horizon < 1:
            raise ValueError(f"min_horizon must be >= 1, got {min_horizon}")
        if max_horizon < min_horizon:
            raise ValueError(f"max_horizon ({max_horizon}) must be >= min_horizon ({min_horizon})")

        self._min_horizon = int(min_horizon)
        self._max_horizon = int(max_horizon)
        self._knn_k       = cfg["knn_k"]
        self._knn_tau     = cfg["knn_tau"]
        self._interp_gamma = cfg["interp_gamma"]
        self._pos_weight  = cfg["pos_weight"]
        self._ang_weight  = cfg["ang_weight"]
        self._grip_weight = cfg["grip_weight"]

        data = np.load(data_path, allow_pickle=True).item()
        eps  = sorted(data.keys())
        data = {
            ep: {k: torch.as_tensor(v, dtype=torch.float32, device=self._device)
                 for k, v in ep_dict.items()}
            for ep, ep_dict in data.items()
        }

        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in eps], device=self._device)
        self._lengths = lengths
        T   = int(lengths.max())
        pad = cfg["pad"]

        def pad_last(key, d):
            out = torch.zeros((len(eps), T, d), device=self._device)
            for i, e in enumerate(eps):
                x = data[e][key]
                if pad and len(x) < T:
                    x = torch.cat([x, x[-1:].repeat(T - len(x), 1)], dim=0)
                out[i, :len(x)] = x
            return out

        self._obs_pos    = pad_last("obs.fingertip_pos", 3)
        self._obs_quat   = pad_last("obs.fingertip_quat", 4)
        self._obs_grip   = pad_last("obs.gripper", 1)
        self._obs_linvel = pad_last("obs.ee_linvel_fd", 3)
        self._obs_angvel = pad_last("obs.ee_angvel_fd", 3)
        self._obs_rel_held  = pad_last("obs.fingertip_pos_rel_held", 3)
        self._obs_rel_fixed = pad_last("obs.fingertip_pos_rel_fixed", 3)

        self._act_pos  = pad_last("action.fingertip_pos", 3)
        self._act_quat = pad_last("action.fingertip_quat", 4)
        self._act_grip = pad_last("action.gripper", 1)
        self._mask     = torch.arange(T, device=self._device).expand(len(eps), T) < lengths[:, None]

        self._num_envs    = num_envs
        self._horizon_env = torch.full((num_envs,), self._max_horizon, dtype=torch.long, device=self._device)

        self._queued     = None
        self._queued_idx = None
        self._q_ptr = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self._q_len = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        self._total_episodes     = len(eps)
        self._max_episode_length = T
        print(f"Loaded {len(eps)} episodes; max length {T} on {self._device}. "
              f"Horizon in [{self._min_horizon}, {self._max_horizon}].")

        self._replay_mode = bool(replay_mode)
        self._replay_ptr  = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        self._prev_last_action = torch.zeros((num_envs, 8), device=self._device)
        self._has_prev_last    = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)

    # --- public helpers ---

    def get_total_episodes(self):
        return self._total_episodes

    def get_max_episode_length(self):
        return self._max_episode_length

    def get_max_per_episode_length(self):
        return self._lengths

    def replay_done(self, eidx: torch.Tensor) -> torch.Tensor:
        """Return a bool tensor: True for envs whose replay trajectory has been fully consumed."""
        L = self._lengths[eidx]  # per-env episode length
        return self._replay_ptr[:eidx.shape[0]] >= L

    def clear(self, env_ids: torch.Tensor | np.ndarray | list):
        """Clear queues for the given env ids. Horizons are re-sampled at refill time."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._q_ptr[env_ids] = 0
        self._q_len[env_ids] = 0
        self._replay_ptr[env_ids] = 0
        self._has_prev_last[env_ids] = False

    # --- core NN ---

    def _nn_indices(self, eidx, pos, quat=None, grip=None, verbose=False):
        obs_p = self._obs_pos[eidx]
        obs_q = self._obs_quat[eidx]
        obs_g = self._obs_grip[eidx]
        mask  = self._mask[eidx]

        pos_term  = self._pos_weight * torch.norm(obs_p - pos[:, None, :], dim=-1)
        ang_term  = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)

        if quat is not None:
            ang_term = self._ang_weight * torch.rad2deg(quat_geodesic_angle(obs_q, quat[:, None, :]))
        if grip is not None:
            grip_term = self._grip_weight * (obs_g.squeeze(-1) - grip.view(-1, 1)).abs()

        dist = (pos_term + ang_term + grip_term).masked_fill(~mask, float("inf"))
        L = mask.long().sum(dim=1)

        if self._knn_tau <= 0.0:
            t0 = dist.argmin(dim=1)
        else:
            k = min(self._knn_k, dist.shape[1])
            d_k, idx_k = torch.topk(dist, k=k, largest=False, dim=1)
            logits = -d_k / self._knn_tau
            logits = logits - logits.max(dim=1, keepdim=True).values
            probs  = torch.softmax(logits, dim=1)
            j  = torch.multinomial(probs, num_samples=1).squeeze(1)
            t0 = idx_k.gather(1, j[:, None]).squeeze(1)

        if verbose:
            mmean = lambda x: x.masked_fill(~mask, torch.nan).nanmean().item()
            print(f"[NN contrib] pos_cm*10: {mmean(pos_term):.3f}, "
                  f"ang_deg/10: {mmean(ang_term):.3f}, "
                  f"grip_L1*2: {mmean(grip_term):.3f}")
        return t0, L

    @torch.no_grad()
    def get_actions(self,
                    eidx: torch.Tensor,
                    pos: torch.Tensor,
                    quat: torch.Tensor | None = None,
                    grip: torch.Tensor | None = None,
                    verbose: bool = False) -> torch.Tensor:
        N = pos.shape[0]

        if self._replay_mode:
            L = self._lengths[eidx]
            t = torch.minimum(self._replay_ptr[:N], (L - 1).clamp(min=0))
            a_pos  = self._act_pos[eidx, t, :]
            a_quat = self._act_quat[eidx, t, :]
            a_grip = self._act_grip[eidx, t, :]
            out = torch.cat([a_pos, a_quat, a_grip], dim=-1)
            self._replay_ptr[:N] += 1
            return out

        if self._queued is None:
            self._queued = torch.empty((self._num_envs, self._max_horizon, 8), device=self._device)
            self._queued_idx = torch.empty((self._num_envs, self._max_horizon), dtype=torch.long, device=self._device)

        refill = (self._q_ptr >= self._q_len)
        if refill.any():
            ids = refill.nonzero(as_tuple=False).squeeze(-1)

            t0, L = self._nn_indices(
                eidx[ids], pos[ids],
                None if quat is None else quat[ids],
                None if grip is None else grip[ids],
                verbose,
            )

            ar  = torch.arange(self._max_horizon, device=self._device)
            idx = torch.minimum(t0[:, None] + ar[None, :], (L - 1).clamp(min=0)[:, None])

            ap = self._act_pos[eidx[ids]]
            aq = self._act_quat[eidx[ids]]
            ag = self._act_grip[eidx[ids]]

            a = torch.cat([
                torch.gather(ap, 1, idx[..., None].expand(-1, -1, 3)),
                torch.gather(aq, 1, idx[..., None].expand(-1, -1, 4)),
                torch.gather(ag, 1, idx[..., None].expand(-1, -1, 1)),
            ], dim=-1)  # (M, H_max, 8)

            H_env = torch.randint(self._min_horizon, self._max_horizon + 1,
                                  size=(ids.numel(),), device=self._device)

            # Chunk-to-chunk continuity: blend new chunk from prev chunk's last action.
            have_prev = self._has_prev_last[ids]
            if have_prev.any():
                prev = self._prev_last_action[ids].unsqueeze(1).expand(-1, self._max_horizon, -1)
                w    = _interp_weights(self._max_horizon, H_env, self._interp_gamma, self._device)
                w8   = w.unsqueeze(-1)

                a_pos  = (1.0 - w8) * prev[..., 0:3] + w8 * a[..., 0:3]
                a_quat = _slerp(prev[..., 3:7], a[..., 3:7], w)
                a_grip = (1.0 - w8) * prev[..., 7:8] + w8 * a[..., 7:8]
                a_blend = torch.cat([a_pos, a_quat, a_grip], dim=-1)

                a = torch.where(have_prev.view(-1, 1, 1), a_blend, a)

            self._queued[ids]     = a
            self._queued_idx[ids] = idx
            self._horizon_env[ids] = H_env
            self._q_ptr[ids] = 0
            self._q_len[ids] = H_env

            last_step = (H_env - 1).clamp_min(0)
            self._prev_last_action[ids] = a[torch.arange(ids.numel(), device=self._device), last_step, :]
            self._has_prev_last[ids] = True

        step_idx = torch.minimum(self._q_ptr, (self._q_len - 1).clamp(min=0))
        out = self._queued[torch.arange(N, device=self._device), step_idx, :]

        has_data = (self._q_ptr < self._q_len)
        self._q_ptr[has_data] += 1

        return out

    def get_episode_traj(self, eps_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (obs_pos, obs_quat, act_pos, act_quat) for a given episode index."""
        if not (0 <= eps_idx < self._total_episodes):
            raise IndexError(f"Episode index {eps_idx} out of range [0, {self._total_episodes - 1}]")
        T = int(self._lengths[eps_idx].item())
        return (
            self._obs_pos[eps_idx, :T, :].clone(),
            self._obs_quat[eps_idx, :T, :].clone(),
            self._act_pos[eps_idx, :T, :].clone(),
            self._act_quat[eps_idx, :T, :].clone(),
        )

    @torch.no_grad()
    def get_closest_obs_pos(self, eidx, pos, quat=None, grip=None, verbose=False, return_idx=False):
        """For each env, return the obs_pos closest to the query (pos, quat, grip)."""
        t0, _ = self._nn_indices(eidx=eidx, pos=pos, quat=quat, grip=grip, verbose=verbose)
        obs_pos_nn = self._obs_pos[eidx, t0, :]
        return (obs_pos_nn, t0) if return_idx else obs_pos_nn

    @torch.no_grad()
    def get_closest_obs(self, eidx, pos, quat=None, grip=None, verbose=False, return_idx=False):
        """For each env, return the obs (pos, quat, grip) closest to the query."""
        t0, _ = self._nn_indices(eidx=eidx, pos=pos, quat=quat, grip=grip, verbose=verbose)
        pos_nn  = self._obs_pos[eidx, t0, :]
        quat_nn = self._obs_quat[eidx, t0, :]
        grip_nn = self._obs_grip[eidx, t0, :]
        return (pos_nn, quat_nn, grip_nn, t0) if return_idx else (pos_nn, quat_nn, grip_nn)
