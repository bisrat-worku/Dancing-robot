# EDGE on this PC: generating dance samples

Goal of this repo for the user: generate EDGE dances from music, then retarget
them to the Unitree G1 humanoid (via GMR).

## Generate a sample: one command

```bash
./generate_dance.sh custom_music/funkorama.wav               # 30 s dance, ~1.5 min
./generate_dance.sh path/to/song.wav --length 20 --seed 3    # --no_render skips the .mp4
./generate_dance.sh custom_music/funkorama.wav --g1          # + retarget to Unitree G1, ~2.5 min total
./generate_dance.sh custom_music/funkorama.wav --seed 0 --guidance 4   # CFG weight (default 2 = stock EDGE)
```
Every run writes `gen_info.json` (seed, guidance weight, window start slice, audio offset in s).

Output goes to `outputs/<song>_<timestamp>/`:
- `<song>.pkl`: raw EDGE motion. `smpl_poses` (T,72) axis-angle, `smpl_trans` (T,3),
  `full_pose` (T,24,3) joint positions. 30 fps, Z-up, **EDGE's floor is at z = 1 m**.
- `<song>_smplx.npz`: AMASS-style SMPL-X for GMR, with the floor shifted to z = 0.
  This is the file to retarget to G1.
- `test_<song>.mp4`: stick-figure render with audio. `seed.txt` has the seed (empty = random).

With `--g1`, the script also writes (about +1 min):
- `<song>_g1.pkl`: Unitree G1 motion in GMR's format. `fps`=30, `root_pos` (T,3),
  `root_rot` (T,4 **xyzw**), `dof_pos` (T,29), `local_body_pos`/`link_body_list` = None.
- `<song>_g1.mp4`: off-screen MuJoCo render of the G1.
- `<song>_g1_fixed.pkl` / `.mp4`: the same motion after `fix_g1_feet.py`, which removes floor penetration and
  locks planted feet. Same format. **Use this one for physics or tracking.**

## Retarget to G1 (EDGE → GMR, no SMPL-X body model)

```bash
python3 g1/edge_to_g1.py outputs/<run>/            # an existing run: adds <song>_g1.pkl
python3 g1/edge_to_g1.py outputs/<run>/ --video    # + <song>_g1.mp4;  --view = interactive viewer
```
- GMR lives in `/home/bwork/GMR` (git clone, its own `uv sync` venv, Python 3.10). `edge_to_g1.py`
  re-launches itself with `/home/bwork/GMR/.venv/bin/python`, so any `python3` works.
- **The SMPL-X body model files are NOT needed and aren't installed.** GMR's `retarget()` only
  takes per-frame `{joint: (pos, quat_wxyz)}`. `edge_to_g1.py` builds that from EDGE's
  `full_pose` and from world rotations chained from `smpl_poses` (GMR's own convention), and feeds it to
  `GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1", actual_human_height=1.66)`.
  Every run checks that FK from those rotations reproduces EDGE's positions (should be ~0 mm).
- Don't route through `edge_to_smplx.py` + GMR's `smplx_to_robot.py`: that needs the SMPL-X model
  (registration-gated download) and always opens a viewer window.
- EDGE's venv has numpy 2.5 and GMR's has 2.2. Their pickles aren't compatible, so `edge_to_g1.py`
  loads EDGE's .pkl with a small compat unpickler. Keep it.
- Verified on funkorama: root height 0.72–0.85 m, ankles 1–5 cm above the floor, 0 joint-limit
  violations. Peak joint speed is about 32 rad/s, so smooth or filter before tracking on real hardware.

## Fix the feet (g1/fix_g1_feet.py)
`python3 g1/fix_g1_feet.py outputs/<run>/ [--video]` writes `<song>_g1_fixed.pkl` and leaves `_g1.pkl` as it is.
`generate_dance.sh --g1` runs it automatically, which adds about 20 s plus the video. It needs EDGE's `contact`
in the .pkl (runs from 2026-11-07 on).
- Contacts: a foot is planted when EDGE's heel or toe flag is > 0.95. Flags flicker (the median plant is 3 frames), so
  gaps of 3 frames or less are filled and plants shorter than 5 frames are dropped.
- Foot targets (ankle_roll_link):
  - Planted: xy is pinned to the plant's mean (shift capped at 12 cm, faded over 5 frames), and z is set so the
    lowest point of the collision spheres and mesh is at 0.
  - Heel and toe both planted: the foot is also levelled (yaw kept).
  - Free: the foot is only lifted when it would go below z = 0.
- Leg IK: damped least squares on each leg's 6 joints in MuJoCo. Where a straight knee can't reach, the pelvis z
  is lowered by the vertical shortfall (smoothed, capped at 8 cm, 4 passes). Without that cap and with the
  warm start carried between frames, the solve diverged (pelvis −135 cm), so keep both.
- Result on funkorama seed 0:
  - Feet at most 3.4 mm below the floor (was 5.1 cm).
  - Slide while planted 5–8 cm/s mean (was 21–23). Planted heel height 0.3–0.4 cm mean (was 1.2–1.4).
  - Pelvis lowered 1.7 cm mean, 5.9 max.
  - 0 limit and velocity violations, but ankle roll is now within 3° of its ±15° limit in about 17% of frames.
  - Feet and pelvis now deviate from GMR's targets on purpose (feet about 4 cm mean).

## Evaluate a G1 retarget
`python3 g1/eval_g1_motion.py outputs/<run>/ [--g1 g1_fixed]` prints a report and writes `<song>_g1[_fixed]_metrics.json`. It covers
fidelity against GMR's scaled targets, floor penetration of the real foot geometry, floating and skating
while EDGE says a foot is planted, joint limits, velocity against the URDF limits, jumps, and the root.
- `model/diffusion.py` now also saves EDGE's predicted foot contacts as `contact` (T,4:
  l_ankle, r_ankle, l_toe, r_toe; > 0.95 = planted) in the .pkl. Runs made before 2026-11-07 lack it.
- Known findings (funkorama, seed 0): legs and pelvis track within ~1 cm (after removing the fixed
  link offset), arms worst (shoulder limits). Feet go up to 5 cm below the floor, inherited from EDGE's
  low toe joints plus the G1 toe_link sitting 1.5 cm above its sole. Foot skating is EDGE's own:
  its contact labels allow 30 cm/s, and sliding is worse at the 2.5 s slice seams.

Run the script as-is. Don't call `test.py` or `jukemirlib` directly (see pitfalls).
Input must be a `.wav` whose name has no `_`. Convert other formats with
`.venv/bin/ffmpeg -i in.mp3 out.wav`.

Timing: about 1.5 min for a new song (about 75 s for Jukebox features, which are then
cached in `cached_features/<song>/`), and about 1 min 20 s for later runs on the same song.
If a run takes much longer, something is wrong. Check for a stray `wget` (see below).

## Environment (already set up; don't reinstall)
- Python: `.venv/bin/python` (3.14, uv venv). Install with `VIRTUAL_ENV=.venv uv pip install ...`.
  torch 2.15.0+cu132 lives inside `.venv` (it overrides `/opt/torch-env` via `torch_env.pth`).
- `pytorch3d` in the venv is a minimal pure-Python stand-in (rotation conversions and
  `RotateAxisAngle`), which covers everything EDGE uses.
- jukebox and jukemirlib were installed with `--no-deps` (their old pins break things), using the
  build constraint `setuptools<70`. Extra runtime deps added by hand: wget, fire, unidecode.
- ffmpeg is `.venv/bin/ffmpeg` (a symlink to imageio-ffmpeg's binary). There's no system ffmpeg, and sudo needs a password.
- Hardware: RTX 5060 Laptop **8 GB VRAM, 7 GB RAM**, WSL2.

## Weights
- `checkpoint.pt` (EDGE model, 1.2 GB) in the repo root.
- `~/.cache/jukemirlib/vqvae.pth.tar` and `prior_level_2.pth.tar` (10.3 GB Jukebox prior).
- `~/.cache/jukemirlib/prior_shards/` (base.pt plus layer_00..71.pt, 9.6 GB), made by
  `data/audio_extraction/shard_jukebox_prior.py`. `generate_dance.sh` rebuilds them if they're missing.

## Pitfalls already solved (don't regress)
- **Stock jukemirlib needs ≥13 GB VRAM**, so features come from
  `data/audio_extraction/jukebox_lowmem.py`, which streams one transformer layer at a time from the shards.
- **Jukebox silently re-downloads the 10 GB prior** to `~/.cache/jukebox/` unless the
  hparams get `restore_vqvae=""` / `restore_prior=""`. jukebox_lowmem.py passes these. This cost 20 min once.
- Jukebox calls `dist.get_rank()`, so jukebox_lowmem.py starts a 1-process gloo group.
- torch ≥2.6 `weights_only` loading: `EDGE.py` allowlists `Normalizer` / `MinMaxScaler`
  via `torch.serialization.safe_globals`. Don't switch to `weights_only=False`.
- `test.py` uses every song folder in `--feature_cache_dir`. The script points it at a
  temp folder holding just the requested song.
- Dance length needs `int(length/2.5)-1` five-second slices. The script caps the length to fit the song.

## Train a G1 tracking policy (mjlab)
```bash
./train_g1_tracking.sh outputs/<run>/                    # reference + PPO training (TensorBoard logs)
./train_g1_tracking.sh custom_music/song.wav             # dance + G1 retarget + training in one go
./train_g1_tracking.sh outputs/<run>/ --play             # viewer on the newest checkpoint
```
- mjlab lives in `/home/bwork/mjlab` (git clone, `uv sync --extra cu130`: the RTX 5060 is sm_120 and needs
  CUDA 13 torch). Its CLIs are `.venv/bin/train`, `play`, `list-envs` (hyphen, not underscore).
- `g1/edge_to_mjlab.py` (re-launches itself in mjlab's venv) turns `<song>_g1_fixed.pkl` into `<song>_mjlab.csv`
  and `<song>_mjlab_motion.npz`, the reference for `--env.commands.motion.motion-file`. It is mjlab's
  `csv_to_npz.py` without the forced W&B upload (`--upload NAME` to opt in). It applies an 8 Hz low-pass by
  default (peak joint speed 32 → 22 rad/s on funkorama). GMR's and mjlab's G1 have the same joint order
  and identical FK. mjlab's foot capsules sit at the floor for the fixed retarget (median +0.1 cm).
- Logs and checkpoints go to `<run>/rl/g1_tracking/<timestamp>_<song>/`. mjlab's own default logger is
  wandb; the script passes tensorboard unless `--wandb`.
- Measured: 4096 envs = 1.9 GB VRAM and 1.45 s per iteration (1024 envs = 0.8 s per iteration). System
  RAM peaks at about 5–6 GB of 7, so close other heavy apps while training.
