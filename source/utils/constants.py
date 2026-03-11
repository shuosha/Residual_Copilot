"""Shared constants for residual copilot scripts."""

PILOT_NAME_MAP = {
    "LaggyPilot":    ("laggy", "bc_expert"),
    "NoisyPilot":    ("noisy", "bc_expert"),
    "ExpertPilot":   ("none",  "bc_expert"),
    "BCPilot":       ("none",  "bc_teleop"),
    "kNNPilot":      ("none",  "knn"),
    "ReplayPilot":   ("none",  "replay"),
}
