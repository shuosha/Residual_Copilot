#!/bin/bash
# Generate side-by-side demo videos comparing ResidualCopilot vs Replay baseline.
#
# Usage:
#   bash scripts/demo_side_by_side.sh <task> [--no-clean] [--headless]
#
# Arguments:
#   task       — GearMesh, PegInsert, or NutThread
#
# Options:
#   --headless   Run without the Isaac Sim viewer (for machines without a display)
#   --no-clean   Keep all intermediate files (rollout frames, single videos)
#
# Output:
#   logs/demos/<task>/demo_*.mp4
#
# Examples:
#   bash scripts/demo_side_by_side.sh GearMesh
#   bash scripts/demo_side_by_side.sh NutThread --no-clean
#   bash scripts/demo_side_by_side.sh GearMesh --headless

set -euo pipefail

# NOTE: Preload system libgomp to avoid "cannot allocate memory in static TLS
# block" from sklearn's bundled copy when running under Isaac Sim.
export LD_PRELOAD="/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+ $LD_PRELOAD}"

# Auto-accept the Omniverse EULA so the interactive prompt doesn't block
# (process substitution in _run disconnects stdin from the TTY).
export OMNI_KIT_ACCEPT_EULA=Y

_run() { "$@"; }

# ── Parse arguments ──────────────────────────────────────────────────────────
TASK=""
NUM_ENVS=20
CLEAN=true
HEADLESS=false

for arg in "$@"; do
    case "$arg" in
        --no-clean) CLEAN=false ;;
        --headless) HEADLESS=true ;;
        GearMesh|PegInsert|NutThread) TASK="$arg" ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "Usage: bash scripts/demo_side_by_side.sh <task> [--no-clean] [--headless]"
    echo "  task: GearMesh, PegInsert, or NutThread"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

HEADLESS_FLAG=()
if $HEADLESS; then
    HEADLESS_FLAG=(--headless)
fi

DEMO_DIR="$PROJECT_DIR/logs/demos/$TASK"
COPILOT_DIR="$PROJECT_DIR/logs/rollouts/eval_${TASK}_with_ResidualCopilot_and_kNNPilot_no_rand"
REPLAY_DIR="$PROJECT_DIR/logs/rollouts/eval_${TASK}_with_ReplayPilot_no_rand"

mkdir -p "$DEMO_DIR"

# ── Run play.py twice ────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo "  DEMO: $TASK  (num_envs=$NUM_ENVS, headless=$HEADLESS)"
echo "════════════════════════════════════════════════════"

CREATED_COPILOT=false
CREATED_REPLAY=false

echo ""
if [[ -d "$COPILOT_DIR" ]]; then
    echo "[1/5] Copilot rollout already exists, reusing: $COPILOT_DIR"
else
    echo "[1/5] Recording ResidualCopilot + kNNPilot..."
    CREATED_COPILOT=true
    _run python "$SCRIPT_DIR/play.py" \
        --task "$TASK" \
        --pilot kNNPilot \
        --copilot ResidualCopilot \
        --num_envs "$NUM_ENVS" \
        --record \
        --no_rand \
        "${HEADLESS_FLAG[@]}"
fi

echo ""
if [[ -d "$REPLAY_DIR" ]]; then
    echo "[2/5] Replay rollout already exists, reusing: $REPLAY_DIR"
else
    echo "[2/5] Recording Replay baseline (no copilot)..."
    CREATED_REPLAY=true
    _run python "$SCRIPT_DIR/play.py" \
        --task "$TASK" \
        --pilot ReplayPilot \
        --num_envs "$NUM_ENVS" \
        --record \
        --no_rand \
        "${HEADLESS_FLAG[@]}"
fi

# Hardcoded episode indices per task (6 best demos each).
case "$TASK" in
    GearMesh)  EPS_LIST=(2 13 3 6 10 17) ;;
    PegInsert) EPS_LIST=(4 6 8 10 11 13) ;;
    NutThread) EPS_LIST=(0 5 10 11 14 16) ;;
esac

# ── Generate single videos (only for selected episodes) ───────────────────
echo ""
echo "[3/5] Generating annotated single videos for copilot run..."
python "$SCRIPT_DIR/vis/to_videos.py" "$COPILOT_DIR" --single --annotate --episodes "${EPS_LIST[@]}"

echo ""
echo "[4/5] Generating annotated single videos for replay run..."
python "$SCRIPT_DIR/vis/to_videos.py" "$REPLAY_DIR" --single --annotate --episodes "${EPS_LIST[@]}"

# ── Stitch side-by-side ─────────────────────────────────────────────────────
echo ""
echo "[5/5] Stitching side-by-side videos..."

COPILOT_VID_DIR="$COPILOT_DIR/videos"
REPLAY_VID_DIR="$REPLAY_DIR/videos"

for i in "${!EPS_LIST[@]}"; do
    eps_idx="${EPS_LIST[$i]}"
    demo_num=$((i + 1))
    eps_tag="$(printf 'eps_%04d' "$eps_idx")"

    copilot_vid="$(ls "$COPILOT_VID_DIR"/*_${eps_tag}_*_annotated.mp4 2>/dev/null | head -1)"
    replay_vid="$(ls "$REPLAY_VID_DIR"/*_${eps_tag}_*_annotated.mp4 2>/dev/null | head -1)"

    if [[ -z "$copilot_vid" || -z "$replay_vid" ]]; then
        echo "  [skip] Missing video for $eps_tag (demo_${demo_num})"
        continue
    fi

    out_path="$DEMO_DIR/demo_${demo_num}.mp4"

    # Stack side-by-side; pad the shorter video by holding its last frame.
    dur_left="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$copilot_vid")"
    dur_right="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$replay_vid")"
    max_dur="$(python3 -c "print(max($dur_left, $dur_right))")"

    pad_left="$(python3 -c "print(max(0, $max_dur - $dur_left))")"
    pad_right="$(python3 -c "print(max(0, $max_dur - $dur_right))")"

    # Colors matching to_videos.py annotation (BGR → RGB hex):
    #   SOFT_RED  (87,110,237) → #ED6E57  — base action
    #   PINK      (193,131,163) → #A383C1 — residual action
    #   SOFT_BLUE (248,159,71) → #479FF8  — net action
    ffmpeg -y -loglevel warning \
        -i "$replay_vid" \
        -i "$copilot_vid" \
        -filter_complex "
            [0:v]tpad=stop_mode=clone:stop_duration=${pad_right},
                 drawtext=text='Direct Teleop':fontsize=24:fontcolor=white:borderw=2:bordercolor=black:x=10:y=10[left];
            [1:v]tpad=stop_mode=clone:stop_duration=${pad_left},
                 drawtext=text='Assisted Teleop':fontsize=24:fontcolor=white:borderw=2:bordercolor=black:x=10:y=10[right];
            [left][right]hstack=inputs=2,
                 drawtext=text='■':fontsize=18:fontcolor=#ED6E57:x=(w/2)-200:y=h-30:borderw=0,
                 drawtext=text='Base Action':fontsize=16:fontcolor=white:borderw=1:bordercolor=black:x=(w/2)-182:y=h-28,
                 drawtext=text='■':fontsize=18:fontcolor=#A383C1:x=(w/2)-68:y=h-30:borderw=0,
                 drawtext=text='Residual Action':fontsize=16:fontcolor=white:borderw=1:bordercolor=black:x=(w/2)-50:y=h-28,
                 drawtext=text='■':fontsize=18:fontcolor=#479FF8:x=(w/2)+100:y=h-30:borderw=0,
                 drawtext=text='Net Action':fontsize=16:fontcolor=white:borderw=1:bordercolor=black:x=(w/2)+118:y=h-28
                 [out]
        " \
        -map "[out]" \
        -c:v libx264 -pix_fmt yuv420p \
        "$out_path"

    echo "  [ok] demo_${demo_num} (${eps_tag}) → $out_path"
done

# ── Cleanup (only delete rollouts we created, not pre-existing ones) ──────────
if $CLEAN; then
    echo ""
    echo "Cleaning intermediate files..."
    $CREATED_COPILOT && rm -rf "$COPILOT_DIR"
    $CREATED_REPLAY && rm -rf "$REPLAY_DIR"
fi

echo ""
echo "Done! Demo videos in: $DEMO_DIR"
