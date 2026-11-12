#!/usr/bin/env bash
# EDGE dance -> Unitree G1 tracking policy with mjlab (BeyondMimic-style motion tracking RL).
#
#   ./train_g1_tracking.sh outputs/<run>/                 # train on an existing run
#   ./train_g1_tracking.sh custom_music/song.wav          # generate dance + G1 retarget first, then train
#   ./train_g1_tracking.sh outputs/<run>/ --play          # watch the latest policy of that run
#
# Pipeline (each step is skipped when its output already exists):
#   1. dance        generate_dance.sh --g1 --no_render     (only when given a .wav)
#   2. retarget     g1/edge_to_g1.py + g1/fix_g1_feet.py        (only when <song>_g1_fixed.pkl is missing)
#   3. reference    g1/edge_to_mjlab.py -> <song>_mjlab_motion.npz (low-pass + resample to 50 Hz + body states)
#   4. train        mjlab `train Mjlab-Tracking-Flat-Unitree-G1`, logs/checkpoints in <run>/rl/
#
# Options:
#   --envs N          parallel environments (default 4096: ~1.9 GB VRAM, ~1.45 s per iteration on this GPU)
#   --iters N         PPO iterations (default 10000; ~1 policy update each, checkpoint every 500)
#   --smooth-hz F     low-pass cutoff for the reference, 0 = off (default 8). Changing it reconverts.
#   --no-state-est    train the policy without base position / linear velocity inputs (easier on hardware)
#   --wandb           log to Weights & Biases (needs `wandb login`); default is TensorBoard in <run>/rl/
#   --resume          continue from the newest checkpoint in <run>/rl/
#   --reconvert       rebuild the reference npz even if it exists
#   --length S / --seed N   passed to generate_dance.sh when the input is a .wav
#   --play            don't train: open the viewer on the latest checkpoint (or --checkpoint PATH)
#   --video           with --play: also record an mp4 to <run>/rl/.../videos/play/
#   -- ARGS...        anything after `--` goes straight to mjlab's train/play
#                     e.g. -- --agent.algorithm.learning-rate 5e-4 --env.rewards.motion-body-pos.weight 2.0
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MJLAB_DIR="${MJLAB_DIR:-/home/bwork/mjlab}"
MJ="$MJLAB_DIR/.venv/bin"
[ -x "$MJ/train" ] || { echo "mjlab not installed at $MJLAB_DIR (cd $MJLAB_DIR && uv sync --extra cu130)"; exit 1; }

[ $# -ge 1 ] || { sed -n 2,30p "$0"; exit 1; }
INPUT="$1"; shift
ENVS=4096; ITERS=10000; SMOOTH=8; TASK=Mjlab-Tracking-Flat-Unitree-G1; LOGGER=tensorboard
RESUME=0; RECONVERT=0; PLAY=0; VIDEO=0; CKPT=latest; GEN_ARGS=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --envs) ENVS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --smooth-hz) SMOOTH="$2"; RECONVERT=1; shift 2 ;;
    --no-state-est) TASK=Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation; shift ;;
    --wandb) LOGGER=wandb; shift ;;
    --resume) RESUME=1; shift ;;
    --reconvert) RECONVERT=1; shift ;;
    --length|--seed) GEN_ARGS+=("$1" "$2"); shift 2 ;;
    --play) PLAY=1; shift ;;
    --checkpoint) CKPT="$(readlink -f "$2")"; shift 2 ;;
    --video) VIDEO=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

# 1. dance from a .wav
if [[ "$INPUT" == *.wav ]]; then
  echo "== [1/4] generating dance + G1 retarget from $INPUT"
  LOG=$(mktemp)
  ./generate_dance.sh "$INPUT" --g1 --no_render "${GEN_ARGS[@]}" | tee "$LOG"
  RUN=$(sed -n 's/^done: //p' "$LOG" | tail -1); rm -f "$LOG"
  [ -n "$RUN" ] || { echo "couldn't find the output folder of generate_dance.sh"; exit 1; }
else
  RUN="$INPUT"
fi
RUN="$(readlink -f "$RUN")"
[ -d "$RUN" ] || { echo "no such run folder: $RUN"; exit 1; }
EDGE_PKL=$(ls "$RUN"/*.pkl 2>/dev/null | grep -v "_g1" | head -1 || true)
[ -n "$EDGE_PKL" ] || { echo "no EDGE .pkl in $RUN"; exit 1; }
SONG=$(basename "$EDGE_PKL" .pkl)

# 2. G1 retarget + foot fix
if [ ! -f "$RUN/${SONG}_g1_fixed.pkl" ]; then
  echo "== [2/4] retargeting to G1 + fixing feet"
  [ -f "$RUN/${SONG}_g1.pkl" ] || python3 g1/edge_to_g1.py "$RUN"
  python3 g1/fix_g1_feet.py "$RUN"
else
  echo "== [2/4] G1 retarget present: ${SONG}_g1_fixed.pkl"
fi

# 3. mjlab reference
NPZ="$RUN/${SONG}_mjlab_motion.npz"
if [ "$RECONVERT" = 1 ] || [ ! -f "$NPZ" ] || [ "$RUN/${SONG}_g1_fixed.pkl" -nt "$NPZ" ]; then
  echo "== [3/4] building mjlab reference (smooth ${SMOOTH} Hz)"
  python3 g1/edge_to_mjlab.py "$RUN" --which g1_fixed --smooth_hz "$SMOOTH"
else
  echo "== [3/4] mjlab reference present: $(basename "$NPZ")  (--reconvert to rebuild)"
fi

LOG_ROOT="$RUN/rl"
cd "$MJLAB_DIR"   # mjlab writes a few relative paths (wandb/, videos) next to the cwd

# 4a. play
if [ "$PLAY" = 1 ]; then
  echo "== playing $TASK on $SONG (checkpoint: $CKPT)"
  ARGS=(--motion-file "$NPZ" --checkpoint-file "$CKPT" --log-root "$LOG_ROOT" --num-envs 1)
  [ "$VIDEO" = 1 ] && ARGS+=(--video)
  exec "$MJ/play" "$TASK" "${ARGS[@]}" "${EXTRA[@]}"
fi

# 4b. train
echo "== [4/4] training $TASK on $SONG: $ENVS envs x $ITERS iterations, logger=$LOGGER"
echo "   logs + checkpoints: $LOG_ROOT/g1_tracking/"
[ "$LOGGER" = tensorboard ] && echo "   watch: $MJ/tensorboard --logdir $LOG_ROOT"
ARGS=(--env.commands.motion.motion-file "$NPZ"
      --env.scene.num-envs "$ENVS"
      --agent.max-iterations "$ITERS"
      --agent.run-name "$SONG"
      --agent.logger "$LOGGER"
      --log-root "$LOG_ROOT")
[ "$RESUME" = 1 ] && ARGS+=(--agent.resume True)
"$MJ/train" "$TASK" "${ARGS[@]}" "${EXTRA[@]}"

echo
echo "done. watch it:  ./train_g1_tracking.sh $RUN --play"
