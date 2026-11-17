"""Four checks on where the clean-run tracking error comes from.

  python3 policy_eval/error_diagnostics.py outputs/<run>/eval/<train-dir>/

Needs raw/clean.npz and raw/clean_repeats.npz (evaluate.py) and raw/clean_start250.npz /
raw/clean_start500.npz:
  python3 policy_eval/rollout.py clean --run outputs/<run> --train_dir <dir> --checkpoint <ckpt> \
      --seeds 0..2 --start_frame 250 --out <eval>/raw/clean_start250.npz      (and 500)

1. Speed: per-frame error of each body against its limb's reference joint speed (RMS of |qd_ref| over the
   limb's joints, as plot_error_vs_speed.py): Pearson r at lag 0 and at the best lag within +-200 ms,
   Spearman rho, and the mean error on the fastest 10 % of frames against the other 90 %
   (Mann-Whitney U, two-sided). Repeated on each of the 5 frame-0 clean runs; the table shows run 0 and the
   range of the fast/rest ratio over the 5.
2. Start: clean runs started at frames 0, 250 (5 s) and 500 (10 s) from the exact reference state. Error
   is compared at the same REFERENCE time. A reset transient follows the start (high error in the first
   seconds after every start); motion-driven error stays at the same reference time whatever the start.
3. Contact: EDGE's predicted contact (funkorama.pkl 'contact', 30 fps, > 0.95 = planted, heel or toe;
   no gap filling) mapped to the 50 fps reference by time. Reference foot slip speed = horizontal speed of
   the reference ankle_roll_link (body_lin_vel_w xy). Ankle error against slip speed on planted frames.
4. Clipping: the reference conversion (g1/edge_to_mjlab.py: 12 Hz low-pass at 30 fps, then np.clip to the
   MuJoCo joint ranges) is repeated without the clip to find the clipped 30 fps frames and joints. Speed
   spikes = local maxima of a limb's reference joint speed above that limb's 99th percentile. A spike
   "falls on" a clipped frame if a clipped frame of the SAME limb is within +-1 frame at 30 fps (+-33 ms).
   EDGE slice seams (multiples of 2.5 s) are checked the same way.
Writes plots/diagnostics_start.png, plots/diagnostics_contact.png, plots/diagnostics_clipping.png and
diagnostics.json. Re-launches itself in mjlab's venv.
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MJLAB_PY = Path("/home/bwork/mjlab/.venv/bin/python")
try:
  import mjlab  # noqa: F401
except ImportError:
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import evaluate as E  # noqa: E402  (Agg backend)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

import metrics as M  # noqa: E402
from plot_error_vs_speed import COL, GRID, INK, INK2, LIMBS, SURF, style  # noqa: E402

LEGS = ["left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"]
EDGE_PY = E.EDGE_PY


def errors_for(ep, ref, bodies, tracked, start=0):
  n = len(ep["body_pos"])
  te = M.tracking_errors(E.slice_motion(ref, start, start + n), E.as_motion(ep, ref["fps"]),
                         [bodies.index(b) for b in tracked], bodies.index("torso_link"))
  return te["pos_local_cm"]  # (n, bodies); row k = reference frame start + k


def lagged_r(e, s, L):
  a, b = (e[L:], s[:len(s) - L]) if L >= 0 else (e[:L], s[-L:])
  return float(np.corrcoef(a, b)[0, 1])


def limb_of(body):
  return next(l for l, (_, bl) in LIMBS.items() if body in bl)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("eval_dir", type=Path)
  a = ap.parse_args()
  ev = a.eval_dir.resolve()
  raw, plots = ev / "raw", ev / "plots"
  meta, clean = E.load_npz(raw / "clean.npz")
  _, reps = E.load_npz(raw / "clean_repeats.npz")
  runs0 = clean[:1] + reps
  ref = E.load_reference(Path(str(meta["motion_file"])))
  fps = ref["fps"]
  T = len(ref["body_pos"])
  bodies = [str(x) for x in meta["body_names"]]
  joints = [str(x) for x in meta["joint_names"]]
  tracked = [str(x) for x in clean[0]["tracked_body_names"]]
  run_dir = Path(str(meta["motion_file"])).parent
  song = Path(str(meta["motion_file"])).name.split("_mjlab")[0]
  qd = np.abs(ref["joint_vel"])
  limb_joints = {l: [i for i, j in enumerate(joints) if re.match(p, j)] for l, (p, _) in LIMBS.items()}
  spd = {l: np.sqrt((qd[:, ix] ** 2).mean(1)) for l, ix in limb_joints.items()}
  R = {"definitions": __doc__}

  # ------------------------------------------------------------------ 1. speed
  maxL = int(round(0.2 * fps))
  res1 = {}
  for b in tracked:
    l = limb_of(b)
    per_run = []
    for ep in runs0:
      e = errors_for(ep, ref, bodies, tracked)[1:, tracked.index(b)]
      s = spd[l][1:len(e) + 1]
      fast = s >= np.quantile(s, 0.9)
      lags = {L: lagged_r(e, s, L) for L in range(-maxL, maxL + 1)}
      Lb = max(lags, key=lags.get)
      per_run.append({"r_lag0": lags[0], "best_lag_ms": 1000 * Lb / fps, "r_best": lags[Lb],
                      "spearman": float(spearmanr(s, e).statistic),
                      "err_fast_cm": float(e[fast].mean()), "err_rest_cm": float(e[~fast].mean()),
                      "ratio": float(e[fast].mean() / e[~fast].mean()),
                      "mwu_p": float(mannwhitneyu(e[fast], e[~fast]).pvalue)})
    res1[b] = {"limb": l, "run0": per_run[0],
               "ratio_range_5_runs": [min(p["ratio"] for p in per_run), max(p["ratio"] for p in per_run)],
               "r_best_range_5_runs": [min(p["r_best"] for p in per_run), max(p["r_best"] for p in per_run)]}
  R["speed"] = res1
  print("\n1. ERROR vs REFERENCE LIMB JOINT SPEED (clean run 0; ranges over 5 clean runs)")
  print(f"{'body':24s} {'r lag0':>6s} {'r best':>6s} {'lag ms':>6s} {'rho':>5s} {'fast10%':>7s} {'rest':>5s} "
        f"{'ratio':>5s} {'ratio range':>11s} {'MWU p':>8s}")
  for b in tracked:
    r = res1[b]["run0"]
    lo, hi = res1[b]["ratio_range_5_runs"]
    print(f"{b:24s} {r['r_lag0']:+6.2f} {r['r_best']:+6.2f} {r['best_lag_ms']:+6.0f} {r['spearman']:+5.2f} "
          f"{r['err_fast_cm']:7.2f} {r['err_rest_cm']:5.2f} {r['ratio']:5.2f} {lo:5.2f}-{hi:4.2f} {r['mwu_p']:8.1e}")

  # ------------------------------------------------------------------ 2. start time
  starts = {0: runs0}
  for S in (250, 500):
    starts[S] = E.load_npz(raw / f"clean_start{S}.npz")[1]
    assert all(int(ep["start_frame"]) == S for ep in starts[S])
  curves = {}  # start -> (n_runs, T) mean leg error on the reference time axis (nan before the start)
  for S, eps in starts.items():
    c = np.full((len(eps), T), np.nan)
    for i, ep in enumerate(eps):
      e = errors_for(ep, ref, bodies, tracked, S)[:, [tracked.index(b) for b in LEGS]].mean(1)
      c[i, S + 1:S + len(e)] = e[1:]
    curves[S] = c
  win = lambda c, t0, t1: float(np.nanmean(c[:, int(t0 * fps):int(t1 * fps)]))
  res2 = {}
  for S in (0, 250, 500):
    t0 = S / fps
    res2[f"start_{S / fps:g}s"] = {
      "first_1s_after_start": win(curves[S], t0, t0 + 1),
      "first_5s_after_start": win(curves[S], t0, t0 + 5),
      "same_window_from_earlier_starts": {f"start_{S2 / fps:g}s": {"first_1s": win(curves[S2], t0, t0 + 1),
                                                                    "first_5s": win(curves[S2], t0, t0 + 5)}
                                          for S2 in (0, 250, 500) if S2 < S}}
  # agreement of the curves where they overlap, after a 1 s settling window
  ov = {}
  for S2 in (250, 500):
    a0 = np.nanmean(curves[0], 0)[S2 + int(fps):]
    b0 = np.nanmean(curves[S2], 0)[S2 + int(fps):]
    ov[f"0s_vs_{S2 / fps:g}s"] = {"pearson_r": float(np.corrcoef(a0, b0)[0, 1]),
                                  "mean_abs_diff_cm": float(np.mean(np.abs(a0 - b0)))}
  # repeat-to-repeat agreement of frame-0 runs, the noise floor for that comparison
  a0, b0 = curves[0][0, 251:], curves[0][1, 251:]
  ov["0s_run0_vs_run1"] = {"pearson_r": float(np.corrcoef(a0, b0)[0, 1]),
                           "mean_abs_diff_cm": float(np.mean(np.abs(a0 - b0)))}
  res2["curve_agreement_after_1s_settling"] = ov
  R["start"] = res2
  print("\n2. START TIME (mean leg-body error, cm, averaged over runs; windows on the reference time axis)")
  for k, v in res2.items():
    if k.startswith("start_"):
      extra = "  ".join(f"[same window from {kk}: 1 s {vv['first_1s']:.2f}, 5 s {vv['first_5s']:.2f}]"
                        for kk, vv in v["same_window_from_earlier_starts"].items())
      print(f"  {k:10s} first 1 s {v['first_1s_after_start']:.2f}  first 5 s {v['first_5s_after_start']:.2f}  {extra}")
  for k, v in ov.items():
    print(f"  overlap {k}: r = {v['pearson_r']:.2f}, mean |diff| = {v['mean_abs_diff_cm']:.2f} cm")

  fig, axs = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True, facecolor=SURF,
                          gridspec_kw={"height_ratios": [2, 1]})
  t = np.arange(T) / fps
  for i, S in enumerate((0, 250, 500)):
    m = np.nanmean(curves[S], 0)
    style(axs[0])
    axs[0].plot(t, m, color=COL[i], lw=1.3, label=f"start at {S / fps:g} s ({len(starts[S])} runs, mean)")
    if S:
      axs[0].axvline(S / fps, color=COL[i], lw=1, ls=":")
  axs[0].set_ylabel("mean leg-body error\n(local) [cm]", color=INK2, fontsize=9)
  axs[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
  axs[0].set_title("Same reference time, different start: does the early leg error follow the start or the motion?",
                   loc="left", fontsize=11, color=INK)
  style(axs[1])
  legspd = np.sqrt((qd[:, limb_joints["left leg"] + limb_joints["right leg"]] ** 2).mean(1))
  axs[1].fill_between(t, legspd, color="#c3c2b7", lw=0)
  axs[1].plot(t, legspd, color=INK2, lw=0.8)
  axs[1].set_ylabel("ref leg joint\nspeed [rad/s]", color=INK2, fontsize=9)
  axs[1].set_xlabel("reference time [s]", color=INK2)
  fig.tight_layout()
  fig.savefig(plots / "diagnostics_start.png", dpi=110, facecolor=SURF)
  plt.close(fig)

  # ------------------------------------------------------------------ 3. contact
  with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "c.npy"
    subprocess.run([str(EDGE_PY), "-c", "import pickle,numpy as np,sys;"
                    "np.save(sys.argv[2], np.asarray(pickle.load(open(sys.argv[1],'rb'))['contact'], float))",
                    str(run_dir / f"{song}.pkl"), str(f)], check=True)
    contact30 = np.load(f)  # (900, 4): l_ankle, r_ankle, l_toe, r_toe
  idx30 = np.clip(np.round(np.arange(T) / fps * 30).astype(int), 0, len(contact30) - 1)
  c = contact30[idx30] > 0.95
  planted = {"left": c[:, 0] | c[:, 2], "right": c[:, 1] | c[:, 3]}
  res3 = {}
  e0 = errors_for(runs0[0], ref, bodies, tracked)
  fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF, sharey=True)
  for i, side in enumerate(("left", "right")):
    b = f"{side}_ankle_roll_link"
    e = e0[1:, tracked.index(b)]
    slip = 100 * np.linalg.norm(ref["body_lin_vel"][1:len(e) + 1, bodies.index(b), :2], axis=1)  # cm/s
    p = planted[side][1:len(e) + 1]
    hi = p & (slip > 10)
    lo = p & (slip <= 10)
    res3[side] = {"frac_planted": float(p.mean()), "ref_slip_planted_mean_cm_s": float(slip[p].mean()),
                  "ref_slip_planted_p95_cm_s": float(np.percentile(slip[p], 95)),
                  "spearman_err_vs_slip_planted": float(spearmanr(slip[p], e[p]).statistic),
                  "err_planted_cm": float(e[p].mean()), "err_swing_cm": float(e[~p].mean()),
                  "err_planted_slip_gt10_cm": float(e[hi].mean()) if hi.any() else None,
                  "err_planted_slip_le10_cm": float(e[lo].mean()) if lo.any() else None,
                  "n_planted_slip_gt10": int(hi.sum()),
                  "mwu_p_slip_gt10_vs_le10": float(mannwhitneyu(e[hi], e[lo]).pvalue) if hi.sum() > 5 else None}
    ax = axs[i]
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.scatter(slip[~p], e[~p], s=6, color="#c3c2b7", lw=0, alpha=0.6, label="EDGE: swing")
    ax.scatter(slip[p], e[p], s=7, color=COL[i], lw=0, alpha=0.6, label="EDGE: planted")
    ax.axvline(10, color=INK2, lw=0.8, ls=":")
    r = res3[side]
    ax.set_title(f"{side} ankle: planted rho = {r['spearman_err_vs_slip_planted']:+.2f}; "
                 f"error planted {r['err_planted_cm']:.1f} vs swing {r['err_swing_cm']:.1f} cm",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("reference foot horizontal speed [cm/s]", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
  axs[0].set_ylabel("ankle position error (local) [cm]", color=INK2, fontsize=9)
  fig.tight_layout()
  fig.savefig(plots / "diagnostics_contact.png", dpi=110, facecolor=SURF)
  plt.close(fig)
  R["contact"] = res3
  print("\n3. CONTACT (clean run 0; EDGE planted = heel or toe flag > 0.95)")
  for side, r in res3.items():
    print(f"  {side}: planted {r['frac_planted']:.0%} of frames; ref slip while planted mean "
          f"{r['ref_slip_planted_mean_cm_s']:.1f}, p95 {r['ref_slip_planted_p95_cm_s']:.1f} cm/s; "
          f"rho(error, slip | planted) {r['spearman_err_vs_slip_planted']:+.2f}; error planted "
          f"{r['err_planted_cm']:.2f} vs swing {r['err_swing_cm']:.2f} cm; planted & slip>10: "
          f"{r['err_planted_slip_gt10_cm']:.2f} cm (n={r['n_planted_slip_gt10']}) vs <=10: "
          f"{r['err_planted_slip_le10_cm']:.2f} cm, MWU p {r['mwu_p_slip_gt10_vs_le10']:.1e}")

  # ------------------------------------------------------------------ 4. clipping
  sys.path.insert(0, str(E.EDGE_ROOT / "g1"))
  from edge_to_mjlab import G1_JOINTS, smooth
  assert G1_JOINTS == joints, "joint order differs from the rollout's"
  g1 = pickle.load(open(run_dir / f"{song}_g1_fixed.pkl", "rb"))
  f30 = float(g1["fps"])
  _, _, dof = smooth(np.asarray(g1["root_pos"], float), np.asarray(g1["root_rot"], float),
                     np.asarray(g1["dof_pos"], float), f30, 12.0)
  lo_, hi_ = meta["joint_range"][:, 0], meta["joint_range"][:, 1]
  over = np.maximum(dof - hi_, 0) + np.maximum(lo_ - dof, 0)  # (T30, 29) rad beyond the range
  clip_frames = np.where((over > 0).any(1))[0]
  assert len(clip_frames) == 76, f"expected the 76 clipped frames logged at conversion, got {len(clip_frames)}"
  clip_joints = {}
  for j in np.where((over > 0).any(0))[0]:
    fr = np.where(over[:, j] > 0)[0]
    clip_joints[joints[j]] = {"n_frames": int(len(fr)), "max_over_deg": float(np.degrees(over[:, j].max())),
                              "times_s": [round(float(x) / f30, 3) for x in fr]}
  limb_clip = {l: set(np.where((over[:, ix] > 0).any(1))[0].tolist()) for l, ix in limb_joints.items()}
  spikes = []
  for l, s in spd.items():
    thr = np.percentile(s, 99)
    pk = [k for k in range(1, T - 1) if s[k] > thr and s[k] >= s[k - 1] and s[k] >= s[k + 1]]
    for k in pk:
      t_s = k / fps
      k30 = int(round(t_s * f30))
      near = sorted(x for x in limb_clip[l] if abs(x - k30) <= 1)
      j = limb_joints[l][int(np.argmax(qd[k, limb_joints[l]]))]
      seam = round(t_s / 2.5) * 2.5
      spikes.append({"limb": l, "time_s": round(t_s, 3), "speed_rad_s": float(s[k]),
                     "fastest_joint": joints[j], "fastest_joint_speed_rad_s": float(qd[k, j]),
                     "on_clipped_frame": bool(near), "clipped_frames_near": [int(x) for x in near],
                     "nearest_seam_s": seam, "within_33ms_of_seam": abs(t_s - seam) <= 1 / f30 + 1e-9})
  # chance: fraction of all frames (per limb) within +-1 30 fps frame of a clipped frame of that limb
  chance = {}
  for l in LIMBS:
    k30 = np.round(np.arange(T) / fps * f30).astype(int)
    cl = np.array(sorted(limb_clip[l]))
    chance[l] = float(np.mean([cl.size and np.min(np.abs(cl - x)) <= 1 for x in k30])) if cl.size else 0.0
  R["clipping"] = {"n_clipped_frames_30fps": int(len(clip_frames)), "joints": clip_joints,
                   "clipped_times_s": [round(float(x) / f30, 3) for x in clip_frames],
                   "spikes": spikes, "chance_frac_frames_near_clip_per_limb": chance}
  print(f"\n4. CLIPPING: {len(clip_frames)} clipped frames at 30 fps")
  for j, v in clip_joints.items():
    ts = v["times_s"]
    print(f"  {j:26s} {v['n_frames']:3d} frames, up to {v['max_over_deg']:.1f} deg over, "
          f"{ts[0]:.2f}-{ts[-1]:.2f} s")
  print("  speed spikes (local maxima above the limb's 99th percentile):")
  for s in sorted(spikes, key=lambda x: -x["speed_rad_s"]):
    print(f"   {s['time_s']:6.2f} s {s['limb']:9s} {s['speed_rad_s']:5.1f} rad/s ({s['fastest_joint']} "
          f"{s['fastest_joint_speed_rad_s']:.1f})  on clipped frame: {s['on_clipped_frame']}"
          f"{'  near seam ' + str(s['nearest_seam_s']) if s['within_33ms_of_seam'] else ''}")
  print("  chance a random frame is that close to a clipped frame of its limb: " +
        ", ".join(f"{l} {v:.1%}" for l, v in chance.items()))

  fig, axs = plt.subplots(len(LIMBS), 1, figsize=(15, 9), sharex=True, facecolor=SURF)
  for ax, (l, s) in zip(axs, spd.items()):
    style(ax)
    ax.fill_between(np.arange(T) / fps, s, color="#c3c2b7", lw=0)
    ax.plot(np.arange(T) / fps, s, color=INK2, lw=0.8)
    for x in sorted(limb_clip[l]):
      ax.axvspan(x / f30 - 0.5 / f30, x / f30 + 0.5 / f30, color=COL[1], alpha=0.5, lw=0)
    for sp in spikes:
      if sp["limb"] == l:
        ax.plot(sp["time_s"], sp["speed_rad_s"], "v", color=INK, ms=6)
    ax.set_ylabel(f"{l}\n[rad/s]", color=INK2, fontsize=8)
  axs[0].set_title("Reference joint speed per limb; orange = frames clipped to joint limits (that limb), "
                   "triangles = spikes above the limb's 99th percentile", loc="left", fontsize=10, color=INK)
  axs[-1].set_xlabel("reference time [s]", color=INK2)
  fig.tight_layout()
  fig.savefig(plots / "diagnostics_clipping.png", dpi=110, facecolor=SURF)
  plt.close(fig)

  (ev / "diagnostics.json").write_text(json.dumps(R, indent=1, default=float))
  print("\n->", ev / "diagnostics.json", "and plots/diagnostics_*.png")


if __name__ == "__main__":
  main()
