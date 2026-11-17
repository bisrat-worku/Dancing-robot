#!/usr/bin/env bash
# One-command EDGE dance generation on this PC (RTX 5060 8 GB, 7 GB RAM, WSL2).
#
#   ./generate_dance.sh custom_music/funkorama.wav                # 30 s dance
#   ./generate_dance.sh path/to/song.wav --length 20 --seed 3     # options below
#
# Options:
#   --length S    seconds of dance (default 30; capped to what the song allows)
#   --seed N      fixes the random window and diffusion noise (default: random)
#   --no_render   skip the stick-figure .mp4
#   --guidance W  classifier-free guidance weight (default 2 = EDGE's stock value in EDGE.py; 0 = ignore music)
#   --g1          also retarget to Unitree G1 with GMR (g1/edge_to_g1.py) and render a G1 video
#   --out DIR     parent output folder (default outputs/)
#
# Output: outputs/<song>_<timestamp>/
#   <song>.pkl          EDGE motion: smpl_poses (T,72), smpl_trans (T,3), full_pose (T,24,3), 30 fps, Z-up
#   <song>_smplx.npz    AMASS-style SMPL-X file for GMR -> Unitree G1 retargeting
#   *.mp4               stick-figure render with audio (unless --no_render)
#   <song>_g1.pkl/.mp4  G1 motion (GMR format) + robot render (with --g1)
#   <song>_g1_fixed.pkl/.mp4  same, feet fixed by g1/fix_g1_feet.py (no floor penetration, planted feet locked)
#
# Steps, each skipped when already done:
#   1. split the Jukebox prior into per-layer shards (one time, ~35 s)
#   2. Jukebox features for this song, cached in cached_features/<song>/ (~1.5 min first time, then instant)
#   3. EDGE diffusion sampling, saving the motion
#   4. convert the motion to SMPL-X .npz
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY=.venv/bin/python
export PATH="$PWD/.venv/bin:$PATH"        # ffmpeg is symlinked into .venv/bin
export PYTHONWARNINGS=ignore
JUKE_CACHE="$HOME/.cache/jukemirlib"
SHARDS="$JUKE_CACHE/prior_shards"

[ $# -ge 1 ] || { sed -n 2,10p "$0"; exit 1; }
WAV="$(readlink -f "$1")"; shift
LENGTH=30; SEED=""; GUIDANCE=2; RENDER=1; G1=0; OUT_PARENT=outputs
while [ $# -gt 0 ]; do
  case "$1" in
    --length) LENGTH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --no_render) RENDER=0; shift ;;
    --g1) G1=1; shift ;;
    --out) OUT_PARENT="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
[ -f "$WAV" ] || { echo "no such file: $WAV"; exit 1; }
[[ "$WAV" == *.wav ]] || { echo "input must be a .wav (convert with: .venv/bin/ffmpeg -i in.mp3 out.wav)"; exit 1; }
SONG="$(basename "${WAV%.wav}")"
[[ "$SONG" != *_* ]] || { echo "song filename must not contain '_' (EDGE parses names on '_'): rename $SONG"; exit 1; }

# --- preflight: required weights (never auto-download the 10 GB prior) ---
for f in checkpoint.pt "$JUKE_CACHE/vqvae.pth.tar"; do
  [ -f "$f" ] || { echo "missing $f (checkpoint: bash download_model.sh)"; exit 1; }
done

# --- 1. prior shards (one time) ---
if [ ! -f "$SHARDS/base.pt" ] || [ ! -f "$SHARDS/layer_65.pt" ]; then
  [ -f "$JUKE_CACHE/prior_level_2.pth.tar" ] || {
    echo "missing $JUKE_CACHE/prior_level_2.pth.tar (10.3 GB). Download it once with:"
    echo "  wget -O $JUKE_CACHE/prior_level_2.pth.tar https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar"
    exit 1; }
  echo "[1/4] sharding Jukebox prior (one time)"
  $PY data/audio_extraction/shard_jukebox_prior.py
else
  echo "[1/4] prior shards present"
fi

# --- 2. features (cached per song) ---
FEAT="cached_features/$SONG"
n_wav=$( (ls "$FEAT"/*.wav 2>/dev/null || true) | wc -l); n_npy=$( (ls "$FEAT"/*.npy 2>/dev/null || true) | wc -l)
if [ "$n_wav" -gt 0 ] && [ "$n_wav" -eq "$n_npy" ]; then
  echo "[2/4] using cached features $FEAT ($n_npy slices)"
else
  echo "[2/4] extracting Jukebox features for $SONG"
  TMP_MUSIC=$(mktemp -d); ln -s "$WAV" "$TMP_MUSIC/$SONG.wav"
  $PY data/audio_extraction/jukebox_lowmem.py --music_dir "$TMP_MUSIC" --out cached_features
  rm -rf "$TMP_MUSIC"
  n_npy=$(ls "$FEAT"/*.npy | wc -l)
fi

# EDGE needs int(length/2.5)-1 consecutive 5 s slices; cap to what the song has.
MAX_LEN=$(( (n_npy + 1) * 5 / 2 ))
if awk "BEGIN{exit !($LENGTH > $MAX_LEN)}"; then
  echo "      song only allows ${MAX_LEN}s, using that"; LENGTH=$MAX_LEN
fi

# --- 3. generate ---
OUT="$OUT_PARENT/${SONG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
TMP_FEAT=$(mktemp -d); ln -s "$PWD/$FEAT" "$TMP_FEAT/$SONG"   # test.py uses every song in the folder
echo "[3/4] generating ${LENGTH}s dance -> $OUT"
ARGS=(--use_cached_features --feature_cache_dir "$TMP_FEAT" --out_length "$LENGTH"
      --save_motions --motion_save_dir "$OUT" --render_dir "$OUT")
[ "$RENDER" = 1 ] || ARGS+=(--no_render)
$PY - "${ARGS[@]}" <<EOF
import random, sys, runpy
seed = "$SEED"
if seed:
    import numpy, torch
    random.seed(int(seed)); numpy.random.seed(int(seed)); torch.manual_seed(int(seed))
# guidance weight: EDGE.py hard-codes 2; set it on the diffusion object after construction
import EDGE as _E
_init = _E.EDGE.__init__
def _patched(self, *a, **k):
    _init(self, *a, **k)
    self.diffusion.guidance_weight = float("$GUIDANCE")
_E.EDGE.__init__ = _patched
# record the random window test.py picks (first 2.5 s slice -> audio offset)
_ri = random.randint; _picked = []
random.randint = lambda a, b: (_picked.append(_ri(a, b)) or _picked[-1])
sys.argv = ["test.py"] + sys.argv[1:]
runpy.run_path("test.py", run_name="__main__")
import json
json.dump({"seed": seed or None, "guidance_weight": float("$GUIDANCE"), "length_s": float("$LENGTH"),
           "window_start_slice": _picked[0], "audio_offset_s": _picked[0] * 2.5,
           "sampler": "long_ddim_sample, 50 DDIM steps, weight ramp clip(linspace(0, 2w, 50), max w)"},
          open("$OUT/gen_info.json", "w"), indent=1)
EOF
rm -rf "$TMP_FEAT"
mv "$OUT/test_$SONG.pkl" "$OUT/$SONG.pkl"

# --- 4. SMPL-X for GMR ---
echo "[4/4] converting to SMPL-X"
$PY g1/edge_to_smplx.py "$OUT/$SONG.pkl" -o "$OUT/${SONG}_smplx.npz"
echo "$SEED" > "$OUT/seed.txt"

if [ "$G1" = 1 ]; then
  echo "[G1] retargeting to Unitree G1 (GMR)"
  $PY g1/edge_to_g1.py "$OUT/$SONG.pkl" --video   # re-launches itself in /home/bwork/GMR/.venv
  echo "[G1] fixing feet: floor penetration + contact foot locking"
  $PY g1/fix_g1_feet.py "$OUT" --video
fi
echo
echo "done: $OUT"
ls -la "$OUT"
