"""Per-body tracking error over time against the reference's joint speed (clean run).

  python3 policy_eval/plot_error_vs_speed.py outputs/<run>/eval/<train-dir>/

Reads raw/clean.npz (the evaluated clean rollout) and the reference motion it names, and writes into plots/:
  error_vs_joint_speed_time.png     per limb: reference joint speed (top) and that limb's body errors (below),
                                    on the same time axis; two panels, never two y-scales on one
  error_vs_joint_speed_scatter.png  per body: error against its limb's joint speed, Spearman rho and best lag
  error_vs_joint_speed.json         the numbers in the scatter titles
Definitions:
  error        body position error in the training reward frame (pos_local_cm, metrics.tracking_errors),
               the same quantity as the report's "position, local"
  joint speed  RMS over the limb's joints of the REFERENCE joint velocity |qd_ref| (rad/s), per frame
  limbs        left/right leg = hip pitch/roll/yaw, knee, ankle pitch/roll; left/right arm = shoulder
               pitch/roll/yaw, elbow, wrist roll/pitch/yaw; torso = waist yaw/roll/pitch (bodies: pelvis,
               torso_link)
  lag          the shift (within +-1 s) maximising the Pearson correlation of error(t) and speed(t - lag);
               lag > 0 means the error peaks after the speed
Frame 0 (the reset state) is left out, as in the report. Re-launches itself in mjlab's venv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

MJLAB_PY = Path("/home/bwork/mjlab/.venv/bin/python")
try:
  import mjlab  # noqa: F401
except ImportError:  # same venv switch as evaluate.py, but re-launching this file
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import evaluate as E  # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import metrics as M

LIMBS = {
  "left leg": (r"left_(hip|knee|ankle)", ["left_hip_roll_link", "left_knee_link", "left_ankle_roll_link"]),
  "right leg": (r"right_(hip|knee|ankle)", ["right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"]),
  "left arm": (r"left_(shoulder|elbow|wrist)", ["left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link"]),
  "right arm": (r"right_(shoulder|elbow|wrist)", ["right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]),
  "torso": (r"waist_", ["pelvis", "torso_link"]),
}
# reference palette (dataviz skill, categorical slots 1-3: validated for all pairs); position along the
# limb (proximal, middle, distal) keeps the same colour in every panel
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def best_lag(err, spd, max_lag):
  best = (0, -2.0)
  for L in range(-max_lag, max_lag + 1):
    a, b = (err[L:], spd[:len(spd) - L]) if L >= 0 else (err[:L], spd[-L:])
    r = np.corrcoef(a, b)[0, 1]
    if r > best[1]:
      best = (L, r)
  return best


def style(ax):
  ax.set_facecolor(SURF)
  ax.grid(axis="y", color=GRID, lw=0.6)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
  ax.tick_params(colors=INK2, labelsize=8)


def main():
  import re
  ap = argparse.ArgumentParser()
  ap.add_argument("eval_dir", type=Path)
  a = ap.parse_args()
  meta, eps = E.load_npz(a.eval_dir / "raw" / "clean.npz")
  ep = eps[0]
  ref = E.load_reference(Path(str(meta["motion_file"])))
  fps = ref["fps"]
  bodies = [str(x) for x in meta["body_names"]]
  joints = [str(x) for x in meta["joint_names"]]
  tracked = [str(x) for x in ep["tracked_body_names"]]
  n = len(ep["body_pos"])
  te = M.tracking_errors(E.slice_motion(ref, 0, n), E.as_motion(ep, fps), [bodies.index(b) for b in tracked],
                         bodies.index("torso_link"))
  err = te["pos_local_cm"][1:]                      # (T-1, bodies)
  qd = np.abs(ref["joint_vel"][1:n])                # (T-1, 29), reference
  t = np.arange(1, n) / fps
  spd = {}
  for limb, (pat, _) in LIMBS.items():
    idx = [i for i, j in enumerate(joints) if re.match(pat, j)]
    assert idx, limb
    spd[limb] = np.sqrt((qd[:, idx] ** 2).mean(1))
  spd_all = np.sqrt((qd ** 2).mean(1))
  max_lag = int(round(1.0 * fps))
  out = a.eval_dir / "plots"
  out.mkdir(exist_ok=True)

  # ---- time figure: per limb, speed panel above its error panel
  fig, axs = plt.subplots(2 * len(LIMBS), 1, figsize=(15, 2.0 * 2 * len(LIMBS)), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1.5] * len(LIMBS)}, facecolor=SURF)
  emax = float(err.max()) * 1.05
  smax = {l: float(v.max()) * 1.08 for l, v in spd.items()}  # own scale per limb (arms reach ~15, legs ~6)
  for k, (limb, (_, bl)) in enumerate(LIMBS.items()):
    a1, a2 = axs[2 * k], axs[2 * k + 1]
    style(a1), style(a2)
    a1.fill_between(t, spd[limb], color="#c3c2b7", lw=0)
    a1.plot(t, spd[limb], color=INK2, lw=1.0)
    a1.set_ylim(0, smax[limb])
    a1.set_ylabel("ref joint\nspeed [rad/s]", color=INK2, fontsize=8)
    a1.set_title(f"{limb}: reference joint speed (RMS over the {limb}'s joints; own scale), then body position error (shared scale)",
                 loc="left", fontsize=10, color=INK)
    for i, b in enumerate(bl):
      a2.plot(t, err[:, tracked.index(b)], color=COL[i], lw=1.2, label=b.replace("_link", ""))
    a2.set_ylim(0, emax)
    a2.set_ylabel("pos error\n(local) [cm]", color=INK2, fontsize=8)
    a2.legend(loc="upper right", fontsize=8, frameon=False, ncol=len(bl), labelcolor=INK)
  axs[-1].set_xlabel("time in the clip [s]", color=INK2)
  fig.suptitle("Clean run: per-body tracking error against the reference's joint speed",
               x=0.01, ha="left", fontsize=13, color=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.985))
  fig.savefig(out / "error_vs_joint_speed_time.png", dpi=110, facecolor=SURF)
  plt.close(fig)

  # ---- scatter figure: one small multiple per body, its limb's speed on x
  stats = {}
  order = [(limb, b) for limb, (_, bl) in LIMBS.items() for b in bl]
  fig, axs = plt.subplots(len(LIMBS), 3, figsize=(14, 3.0 * len(LIMBS)), facecolor=SURF)
  for k, (limb, (_, bl)) in enumerate(LIMBS.items()):
    for i in range(3):
      ax = axs[k, i]
      if i >= len(bl):
        ax.axis("off")
        continue
      b = bl[i]
      e, s = err[:, tracked.index(b)], spd[limb]
      rho = float(spearmanr(s, e).statistic)
      rho_all = float(spearmanr(spd_all, e).statistic)
      L, rL = best_lag(e, s, max_lag)
      stats[b] = {"limb": limb, "spearman_rho_limb_speed": rho, "spearman_rho_whole_body_speed": rho_all,
                  "best_lag_ms": 1000 * L / fps, "pearson_r_at_best_lag": float(rL),
                  "error_mean_cm": float(e.mean()), "limb_speed_mean_rad_s": float(s.mean())}
      style(ax)
      ax.grid(axis="x", color=GRID, lw=0.6)
      ax.scatter(s, e, s=6, color=COL[i], alpha=0.35, lw=0)
      # binned median: 10 equal-count speed bins
      edges = np.quantile(s, np.linspace(0, 1, 11))
      mid, med = [], []
      for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        mid.append(np.median(s[m])), med.append(np.median(e[m]))
      ax.plot(mid, med, color=INK, lw=2, marker="o", ms=4, label="median per speed decile")
      ax.set_xlim(0, smax[limb])
      ax.set_ylim(0, emax)
      ax.set_title(f"{b.replace('_link', '')}   rho = {rho:+.2f}   lag {1000 * L / fps:+.0f} ms (r = {rL:.2f})",
                   loc="left", fontsize=9, color=INK)
      if i == 0:
        ax.set_ylabel("pos error (local) [cm]", color=INK2, fontsize=8)
      if k == len(LIMBS) - 1 or (k == len(LIMBS) - 2 and i == 2):
        ax.set_xlabel(f"reference joint speed of the limb [rad/s]", color=INK2, fontsize=8)
  axs[0, 0].legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
  fig.suptitle("Clean run: error against the limb's reference joint speed, one dot per frame "
               "(rho = Spearman at lag 0; lag = shift maximising Pearson r, + = error after speed).\n"
               "x axis: each limb on its own scale; y axis shared",
               x=0.01, ha="left", fontsize=11, color=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.975))
  fig.savefig(out / "error_vs_joint_speed_scatter.png", dpi=110, facecolor=SURF)
  plt.close(fig)

  (out / "error_vs_joint_speed.json").write_text(json.dumps(
    {"run": "clean", "fps": fps, "frames": int(n - 1), "bodies": stats}, indent=1))
  print(f"{'body':26s} {'limb':9s} {'err cm':>6s} {'rho limb':>8s} {'rho all':>8s} {'lag ms':>7s} {'r@lag':>6s}")
  for limb, b in order:
    s = stats[b]
    print(f"{b:26s} {limb:9s} {s['error_mean_cm']:6.2f} {s['spearman_rho_limb_speed']:+8.2f} "
          f"{s['spearman_rho_whole_body_speed']:+8.2f} {s['best_lag_ms']:+7.0f} {s['pearson_r_at_best_lag']:6.2f}")
  print("->", out)


if __name__ == "__main__":
  main()
