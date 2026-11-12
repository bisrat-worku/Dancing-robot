# EDGE → Unitree G1: music-driven humanoid dance

Generate a dance from any song with [EDGE](https://github.com/Stanford-TML/EDGE), retarget it to the
Unitree G1 humanoid with [GMR](https://github.com/YanjieZe/GMR), and train a physics-based tracking
policy for it with [mjlab](https://github.com/mujocolab/mjlab) (BeyondMimic-style motion tracking RL).

```
song.wav ──EDGE──▶ SMPL dance ──GMR──▶ G1 joint motion ──foot fix──▶ clean G1 motion ──mjlab PPO──▶ G1 policy (.pt / .onnx)
```

This is a fork of the official EDGE implementation, adapted to run on a single 8 GB laptop GPU
(RTX 5060, WSL2) and extended with a G1 retargeting and RL pipeline.

## Quick start

```bash
# 1. music → dance (+ G1 retarget with --g1)           ~1.5 min (2.5 min with --g1)
./generate_dance.sh custom_music/song.wav --g1

# 2. dance → tracking policy (TensorBoard logs in the run folder)
./train_g1_tracking.sh outputs/<song>_<timestamp>/

# 3. watch the policy in the MuJoCo viewer
./train_g1_tracking.sh outputs/<song>_<timestamp>/ --play
```

`./train_g1_tracking.sh custom_music/song.wav` runs all of it in one go. Input songs must be `.wav`
files without `_` in the name.

## Repository layout

```
generate_dance.sh        music → EDGE dance → SMPL-X (+ G1 with --g1)
train_g1_tracking.sh     dance → mjlab reference → PPO training / --play
g1/
  edge_to_g1.py          EDGE → G1 with GMR, straight from EDGE's joints (no SMPL-X body model needed)
  fix_g1_feet.py         removes floor penetration, locks planted feet (leg IK in MuJoCo)
  eval_g1_motion.py      retarget quality report: fidelity, floor, skating, limits, velocities
  edge_to_smplx.py       EDGE → AMASS-style SMPL-X .npz
  edge_to_mjlab.py       G1 motion → mjlab tracking reference (low-pass, 50 Hz, body states)
EDGE.py, test.py, vis.py, args.py, model/, dataset/   EDGE diffusion model and inference (upstream)
data/slice.py, data/audio_extraction/                  audio slicing and Jukebox features
  jukebox_lowmem.py      streams Jukebox one layer at a time (fits in 8 GB instead of 13+ GB)
  shard_jukebox_prior.py splits the 10 GB Jukebox prior into per-layer shards (one-time)
```

Each run writes to `outputs/<song>_<timestamp>/`:

| File | Content |
|---|---|
| `<song>.pkl` | EDGE motion: SMPL poses, root translation, joint positions, foot contacts (30 fps) |
| `<song>_smplx.npz` | SMPL-X version for other retargeters |
| `<song>_g1.pkl`, `<song>_g1_fixed.pkl` | G1 motion in GMR's format (root pose + 29 joints); use `_fixed` |
| `<song>_mjlab_motion.npz` | mjlab tracking reference (50 Hz) |
| `rl/g1_tracking/<timestamp>_<song>/` | checkpoints (`model_*.pt`), exported `policy.onnx`, TensorBoard logs |

## Setup

External pieces (not in this repo):

| What | Where | Notes |
|---|---|---|
| EDGE checkpoint | `checkpoint.pt` | `bash download_model.sh` |
| Jukebox weights | `~/.cache/jukemirlib/` | `vqvae.pth.tar`, `prior_level_2.pth.tar`; shards are built automatically |
| GMR | `/home/bwork/GMR` (or `$GMR_DIR`) | `uv sync`, Python 3.10 |
| mjlab | `/home/bwork/mjlab` (or `$MJLAB_DIR`) | `uv sync --extra cu130` (CUDA 13 for RTX 50xx) |

The scripts in `g1/` re-launch themselves in the GMR or mjlab venv, so any `python3` works.
See `CLAUDE.md` for environment details, measured results and solved pitfalls.

## Training the tracking policy

`./train_g1_tracking.sh outputs/<run>/ [options]`

| Option | Default | Meaning |
|---|---|---|
| `--envs N` | 4096 | parallel simulations (1.9 GB VRAM, about 1.45 s per iteration on an RTX 5060 Laptop) |
| `--iters N` | 10000 | PPO iterations, checkpoint every 500 |
| `--smooth-hz F` | 8 | low-pass cutoff for the reference, 0 = off |
| `--no-state-est` | off | policy without base position / linear velocity inputs (for real hardware) |
| `--wandb` | off | log to Weights & Biases instead of TensorBoard (`wandb login` first) |
| `--resume` | off | continue from the newest checkpoint |
| `-- ARGS` | | passed to mjlab, e.g. `-- --env.rewards.motion-body-pos.weight 2.0` |

## Credits

Built on **EDGE: Editable Dance Generation From Music** (CVPR 2023), Jonathan Tseng, Rodrigo Castellon,
C. Karen Liu ([paper](https://arxiv.org/abs/2211.10658), [code](https://github.com/Stanford-TML/EDGE)),
which uses [Jukebox](https://github.com/openai/jukebox) features and lucidrains'
[Adan](https://github.com/lucidrains/Adan-pytorch) and
[diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) code.
Retargeting by [GMR](https://github.com/YanjieZe/GMR); RL by [mjlab](https://github.com/mujocolab/mjlab),
a re-implementation of [BeyondMimic](https://beyondmimic.github.io/).

```bibtex
@article{tseng2022edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C Karen},
  journal={arXiv preprint arXiv:2211.10658},
  year={2022}
}
```

MIT License (see `LICENSE`, original copyright Jonathan Tseng).
