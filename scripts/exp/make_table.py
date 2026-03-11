"""Generate a task-progression table from sim_exp results.

Rows    = copilots (5)
Columns = pilots (5), each a multicolumn with 3 task sub-columns (Gear, Peg, Nut)

Task progression = 1 - error_mean / error_max   (1 = perfect, 0 = worst)
Std is propagated as std / error_max.

Usage:
    python scripts/make_table.py \
        --gear logs/sim_exp/GearMesh/results.json \
        --peg  logs/sim_exp/PegInsert/results.json \
        --nut  logs/sim_exp/NutThread/results.json
"""

import argparse
import json
import os

PILOTS = ["LaggyPilot", "NoisyPilot", "ExpertPilot", "BCPilot", "kNNPilot"]
COPILOTS = ["None", "GuidedDiffusionExpert", "GuidedDiffusionBC", "ResidualBC", "ResidualCopilot"]

PILOT_DISPLAY = {
    "LaggyPilot": "Laggy",
    "NoisyPilot": "Noisy",
    "ExpertPilot": "Expert",
    "BCPilot": "BC",
    "kNNPilot": "kNN",
}
COPILOT_DISPLAY = {
    "None": "No Copilot",
    "GuidedDiffusionExpert": "GD (Expert)",
    "GuidedDiffusionBC": "GD (BC)",
    "ResidualBC": "Residual (BC)",
    "ResidualCopilot": "Residual Copilot",
}
TASK_DISPLAY = {"gear": "Gear", "peg": "Peg", "nut": "Nut"}

# error_max per task (from CLIP_VALUES in sim_exp.py)
ERROR_MAX = {
    "gear": 0.15,
    "peg": 0.15,
    "nut": 90.0,
}


def load_results(result_path):
    """Load results.json and return {(copilot, pilot): dict}."""
    results = {}
    if result_path is None or not os.path.isfile(result_path):
        return results
    with open(result_path) as f:
        data = json.load(f)
    for copilot, pilots in data.get("results", {}).items():
        for pilot, metrics in pilots.items():
            results[(copilot, pilot)] = metrics
    return results


def make_table(task_dirs):
    """Build and print a plain-text table."""
    task_keys = ["gear", "peg", "nut"]
    all_data = {}
    for tk in task_keys:
        all_data[tk] = load_results(task_dirs.get(tk))

    active_tasks = [tk for tk in task_keys if task_dirs.get(tk)]

    # Column widths
    copilot_w = max(len(v) for v in COPILOT_DISPLAY.values()) + 2
    cell_w = 12
    task_group_w = cell_w * len(active_tasks)

    # Separator
    sep_parts = ["-" * copilot_w]
    for _ in PILOTS:
        sep_parts.append("-" * task_group_w)
    sep = "+" + "+".join(sep_parts) + "+"

    lines = []

    # Header row 1: pilot names centered over their task group
    h1 = "|" + " " * copilot_w + "|"
    for pilot in PILOTS:
        h1 += PILOT_DISPLAY[pilot].center(task_group_w) + "|"
    lines.append(sep)
    lines.append(h1)

    # Header row 2: task names
    h2 = "|" + "Copilot".center(copilot_w) + "|"
    for _ in PILOTS:
        for tk in active_tasks:
            h2 += TASK_DISPLAY[tk].center(cell_w)
        h2 += "|"
    lines.append(sep)
    lines.append(h2)
    lines.append(sep)

    # Data rows
    for copilot in COPILOTS:
        row = "|" + COPILOT_DISPLAY[copilot].center(copilot_w) + "|"
        for pilot in PILOTS:
            for tk in active_tasks:
                data = all_data[tk].get((copilot, pilot))
                if data is None:
                    cell = "---"
                else:
                    err_mean = data["error_mean"]
                    err_std = data["error_std"]
                    emax = ERROR_MAX[tk]
                    prog_mean = 1.0 - err_mean / emax
                    prog_std = err_std / emax
                    cell = f"{prog_mean:.2f}±{prog_std:.2f}"
                row += cell.center(cell_w)
            row += "|"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate task-progression table.")
    parser.add_argument("--gear", type=str, default=None, help="Path to GearMesh results.json")
    parser.add_argument("--peg", type=str, default=None, help="Path to PegInsert results.json")
    parser.add_argument("--nut", type=str, default=None, help="Path to NutThread results.json")
    args = parser.parse_args()

    if not any([args.gear, args.peg, args.nut]):
        parser.error("Provide at least one of --gear, --peg, --nut")

    task_dirs = {"gear": args.gear, "peg": args.peg, "nut": args.nut}
    print(make_table(task_dirs))


if __name__ == "__main__":
    main()
