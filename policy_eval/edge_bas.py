"""Beat alignment of raw EDGE motions (SMPL joints, 30 fps), with the same scoring as evaluate.py.

  # EDGE runs (folders from generate_dance.sh, each with <song>.pkl and gen_info.json)
  .venv/bin/python policy_eval/edge_bas.py outputs/guidance_sweep/w*_s* --song_dir custom_music \
      --json outputs/guidance_sweep/bas.json
  # plus the real AIST++ dances for the same songs (ground truth control)
  .venv/bin/python policy_eval/edge_bas.py outputs/guidance_sweep_aist/*/w*_s* \
      --song_dir custom_music/aistpp_test --aist_gt /mnt/c/Users/bwork/edge_data/edge_aistpp/motions

Per motion:
  motion       full_pose (T, 24, 3), all 24 SMPL joints, 30 fps. AIST++ ground truth: smpl_poses and
               smpl_trans / smpl_scaling through EDGE's own SMPLSkeleton (vis.py), 60 fps taken every 2nd
               frame (as EDGE's dataset does), first n frames where n = the EDGE motions' length
  music beats  music_beats.beats_for at the motion's audio offset (gen_info.json; 0 for ground truth,
               whose wav starts at motion frame 0), on the 30 fps grid
  kin. beats   minima of the Gaussian-smoothed kinetic velocity (metrics.kinematic_beats)
  scores       kin->music (primary) and music->kin (Bailando/EDGE paper direction), sigma = 0.05 s
  controls     chance = mean over 500 random phase shifts of the music beat grid;
               wrong songs = 3 librosa example tracks + the right song shifted by half a beat (anti-phase)
  test passes  kin->music on the right song > chance mean AND > every wrong song
Runs in EDGE's venv (librosa).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
from config import EvalConfig  # noqa: E402
from music_beats import beats_for  # noqa: E402

WRONG = ("vibeace", "fishin", "nutcracker")
FPS = 30.0


def score_motion(pos, song: Path, off: float, cfg: EvalConfig, fps=FPS):
  import librosa
  T = len(pos)
  sig = cfg.beat_sigma_s * fps
  kb, _ = M.kinematic_beats(pos, fps, cfg.kinetic_smooth_sigma_s)
  mb = beats_for(str(song), off, fps, T, cfg.music_hop_length, cfg.music_beat_tightness)
  right = M.beat_alignment(kb, mb["beat_frames"], sig)
  chance = M.beat_alignment_chance(kb, mb["beat_frames"], sig)
  wrong = {}
  for k in WRONG:
    b = beats_for(librosa.ex(k), 0.0, fps, T, cfg.music_hop_length, cfg.music_beat_tightness)
    wrong[k] = M.beat_alignment(kb, b["beat_frames"], sig)["kin_to_music"]
  half = 0.5 * float(np.mean(np.diff(mb["beat_frames"]))) / fps
  b = beats_for(str(song), off + half, fps, T, cfg.music_hop_length, cfg.music_beat_tightness)
  wrong["self_half_beat"] = M.beat_alignment(kb, b["beat_frames"], sig)["kin_to_music"]
  passed = right["kin_to_music"] > chance["mean"] and all(right["kin_to_music"] > v for v in wrong.values())
  return {"song": song.stem, "audio_offset_s": off, "n_frames": T, "tempo_bpm": mb["tempo_bpm"],
          "n_music_beats": len(mb["beat_frames"]), **right, "chance_kin_to_music": chance["mean"],
          "wrong_kin_to_music": wrong, "wrong_max": max(wrong.values()), "passes": bool(passed),
          "kinetic_vel_mean_m_s": float(M.kinetic_velocity(pos).mean() * fps)}


def score_run(run: Path, song_dir: Path, cfg: EvalConfig):
  info = json.loads((run / "gen_info.json").read_text())
  pkl = next(p for p in run.glob("*.pkl") if "_g1" not in p.name)
  pos = np.asarray(pickle.load(open(pkl, "rb"))["full_pose"], dtype=np.float64)
  r = score_motion(pos, song_dir / f"{pkl.stem}.wav", float(info["audio_offset_s"]), cfg)
  return {"run": str(run), "source": "EDGE", "guidance_weight": info["guidance_weight"],
          "seed": info["seed"], **r}


def gt_joints(aist_pkl: Path):
  import torch
  sys.path.insert(0, str(HERE.parent))
  from vis import SMPLSkeleton
  d = pickle.load(open(aist_pkl, "rb"))
  q = torch.tensor(d["smpl_poses"][::2], dtype=torch.float32).reshape(1, -1, 24, 3)
  t = torch.tensor(d["smpl_trans"][::2] / d["smpl_scaling"][0], dtype=torch.float32)[None]
  return SMPLSkeleton().forward(q, t)[0].numpy().astype(np.float64)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+", type=Path)
  ap.add_argument("--song_dir", type=Path, required=True)
  ap.add_argument("--aist_gt", type=Path, help="AIST++ motions dir; needs <song_dir>/sources.json")
  ap.add_argument("--json", type=Path)
  a = ap.parse_args()
  cfg = EvalConfig()
  rows = [score_run(r, a.song_dir, cfg) for r in sorted(a.runs)]
  if a.aist_gt:
    src = json.loads((a.song_dir / "sources.json").read_text())
    for song in sorted({r["song"] for r in rows}):
      n = min(r["n_frames"] for r in rows if r["song"] == song)
      pos = gt_joints(a.aist_gt / f"{src[song]}.pkl")[:n]
      rows.append({"run": src[song], "source": "AIST++ ground truth", "guidance_weight": None, "seed": None,
                   **score_motion(pos, a.song_dir / f"{song}.wav", 0.0, cfg)})

  key = lambda r: (r["song"], r["guidance_weight"] is None, r["guidance_weight"] or 0, str(r["seed"]))
  rows.sort(key=key)
  print(f"{'song':>6} {'w':>4} {'seed':>4} {'off':>5} {'kin->mus':>8} {'mus->kin':>8} {'chance':>7} "
        f"{'wrongmax':>8} {'pass':>5}")
  for r in rows:
    w = "GT" if r["guidance_weight"] is None else f"{r['guidance_weight']:g}"
    print(f"{r['song']:>6} {w:>4} {str(r['seed'] or '-'):>4} {r['audio_offset_s']:>5g} {r['kin_to_music']:>8.3f} "
          f"{r['music_to_kin']:>8.3f} {r['chance_kin_to_music']:>7.3f} {r['wrong_max']:>8.3f} "
          f"{str(r['passes']):>5}")

  def agg(g, label):
    f = lambda k: np.mean([r[k] for r in g])
    above = np.mean([r["kin_to_music"] - r["chance_kin_to_music"] for r in g])
    beats_wrong = np.mean([np.mean([r["kin_to_music"] > v for v in r["wrong_kin_to_music"].values()]) for r in g])
    print(f"  {label:>12}: n={len(g):>3}  kin->music {f('kin_to_music'):.3f}  music->kin {f('music_to_kin'):.3f}  "
          f"chance {f('chance_kin_to_music'):.3f}  above chance {above:+.3f}  "
          f"beats a wrong song {beats_wrong:.0%}  passes {sum(r['passes'] for r in g)}/{len(g)}")

  print("\nall songs pooled:")
  for w in sorted({r["guidance_weight"] for r in rows if r["guidance_weight"] is not None}):
    agg([r for r in rows if r["guidance_weight"] == w], f"EDGE w={w:g}")
  gt = [r for r in rows if r["guidance_weight"] is None]
  if gt:
    agg(gt, "ground truth")
  if a.json:
    a.json.write_text(json.dumps({"config": {"beat_sigma_s": cfg.beat_sigma_s,
                                             "kinetic_smooth_sigma_s": cfg.kinetic_smooth_sigma_s,
                                             "fps": FPS, "wrong_songs": WRONG + ("self_half_beat",)},
                                  "runs": rows}, indent=1))


if __name__ == "__main__":
  main()
