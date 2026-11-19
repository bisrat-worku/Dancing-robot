"""Compare two evaluated training runs (evaluate.py outputs) and plot the leg-constraint multiplier.

  python3 policy_eval/compare_runs.py --a outputs/<run>/eval/<stock dir> --b outputs/<run>/eval/<constrained dir> \
      [--base outputs/<run>/eval/<0-10k dir>] --out outputs/<run>/eval/compare_<name>

Reads each eval dir's metrics.json and raw/{randomized,push}.npz; nothing is re-simulated. Writes into --out:
  comparison.md / .json        the table: legs / arms / torso / body-mean local position error, global error,
                               completion under randomization and under pushes, joint jerk, action rate,
                               torque saturation; clean run, randomized set (50 seeds) and push set (50 seeds)
  lambda_ebar.png              lambda, e_bar (and e_batch) against iteration with eps marked, plus the all-body
                               training error of both runs (Constraint/e_body vs the stock Metrics/motion/error_body_pos)
  group_error_vs_iteration.png legs / arms / torso clean-run error per checkpoint (evaluate.py's sweep), both runs
Definitions (all from evaluate.py / metrics.py, unchanged):
  local error   pos_local_cm: body position error in the training reward frame; group = mean over the group's
                bodies of the per-body mean (as evaluate.py's sweep), body-mean = mean over all 14 bodies
  global error  pos_global_cm body-mean (world frame, no re-anchoring)
  pooled        randomized / push: every scored frame of the 50 episodes pooled (evaluate.py's "tracking")
  paired test   both runs are evaluated on the same seeds (1000..1049 randomized, push seeds likewise), so
                per-episode values are paired: Wilcoxon signed-rank on (b - a), two-sided
Re-launches itself in mjlab's venv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

MJLAB_PY = Path("/home/bwork/mjlab/.venv/bin/python")
try:
  import mjlab  # noqa: F401
except ImportError:
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import evaluate as E  # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import metrics as M  # noqa: E402
from config import EvalConfig  # noqa: E402

COL = {"a": "#2a78d6", "b": "#eb6834", "base": "#898781"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
GROUPS = EvalConfig().body_groups


def style(ax):
  ax.set_facecolor(SURF)
  ax.grid(axis="y", color=GRID, lw=0.6)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
  ax.tick_params(colors=INK2, labelsize=8)


def per_episode_groups(eval_dir: Path, name: str):
  """Per-episode group means of pos_local_cm (frames 1.. up to termination), same as evaluate.py scores them."""
  p = eval_dir / "raw" / f"{name}.npz"
  meta, eps = E.load_npz(p)
  ref = E.load_reference(Path(str(meta["motion_file"])))
  bodies = [str(x) for x in meta["body_names"]]
  out = []
  for ep in eps:
    tracked = [str(x) for x in ep["tracked_body_names"]]
    n = len(ep["body_pos"])
    s0 = int(ep.get("start_frame", 0)) if hasattr(ep, "get") else 0
    te = M.tracking_errors(E.slice_motion(ref, s0, s0 + n), E.as_motion(ep, ref["fps"]),
                           [bodies.index(b) for b in tracked], bodies.index("torso_link"))
    e = te["pos_local_cm"][1:]
    row = {"seed": int(ep["seed"]), "body_mean": float(e.mean())}
    for g, bs in GROUPS.items():
      row[g] = float(np.mean([e[:, tracked.index(b)].mean() for b in bs]))
    row["global_body_mean"] = float(te["pos_global_cm"][1:].mean())
    out.append(row)
  return out


def summarize(eval_dir: Path):
  m = json.loads((eval_dir / "metrics.json").read_text())
  r = {"checkpoint": m["checkpoint"], "task": m["config"]["task"]}
  for prot in ("clean", "randomized", "push"):
    if prot not in m:
      continue
    tr = m[prot]["tracking"]
    pb = tr["pos_local_cm"]["per_body"]
    d = {g: float(np.mean([pb[b]["mean"] for b in bs])) for g, bs in GROUPS.items()}
    d["body_mean"] = tr["pos_local_cm"]["body_mean"]["mean"]
    d["global_body_mean"] = tr["pos_global_cm"]["body_mean"]["mean"]
    if prot == "clean":
      s = m["clean"]["summary"]
      d.update(completion=s["completion_fraction"], success=float(s["completed"]),
               joint_jerk=s["exec"]["joint_jerk_mean"], action_rate=s["action_rate_l2"]["mean"],
               torque_sat=s["torque_sat_frac_any_joint"],
               repeat_body_mean_range=[m["clean_repeats"]["spread"]["pos_local_cm_mean"]["min"],
                                       m["clean_repeats"]["spread"]["pos_local_cm_mean"]["max"]])
    else:
      s, eps = m[prot]["set"], m[prot]["episodes"]
      d.update(completion=s["completion_fraction"]["mean"], success=s["success_rate"],
               joint_jerk=float(np.mean([e["exec"]["joint_jerk_mean"] for e in eps])),
               action_rate=float(np.mean([e["action_rate_l2"]["mean"] for e in eps])),
               torque_sat=float(np.mean([e["torque_sat_frac_any_joint"] for e in eps])),
               per_episode={"completion": {int(e["seed"]): e["completion_fraction"] for e in eps},
                            "joint_jerk": {int(e["seed"]): e["exec"]["joint_jerk_mean"] for e in eps},
                            "action_rate": {int(e["seed"]): e["action_rate_l2"]["mean"] for e in eps},
                            "torque_sat": {int(e["seed"]): e["torque_sat_frac_any_joint"] for e in eps}})
      for row in per_episode_groups(eval_dir, prot):
        for k, v in row.items():
          if k != "seed":
            d["per_episode"].setdefault(k, {})[row["seed"]] = v
    r[prot] = d
  r["sweep"] = m.get("sweep", [])
  return r


def paired(a, b, key):
  pa, pb = a["per_episode"][key], b["per_episode"][key]
  seeds = sorted(set(pa) & set(pb))
  x, y = np.array([pa[s] for s in seeds]), np.array([pb[s] for s in seeds])
  d = y - x
  p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
  return {"n": len(seeds), "median_diff": float(np.median(d)), "b_lower_frac": float(np.mean(d < 0)), "p": p}


ROWS = [("legs", "legs local error [cm]", 2), ("arms", "arms local error [cm]", 2),
        ("torso", "torso local error [cm]", 2), ("body_mean", "body-mean local error [cm]", 2),
        ("global_body_mean", "body-mean global error [cm]", 2), ("completion", "completion (mean fraction)", 3),
        ("success", "episodes completing the clip", 2), ("joint_jerk", "joint jerk [rad/s³]", 0),
        ("action_rate", "action rate (L2)", 3), ("torque_sat", "torque saturation (frac. frames, any joint)", 4)]


def table(A, B, la, lb):
  lines = []
  for prot, title in (("clean", "Clean run (no randomization, 1 rollout; the 5 repeats' body-mean range is given)"),
                      ("randomized", "Randomized set (50 seeds: startup DR, RSI, obs noise; no pushes)"),
                      ("push", "Push set (50 seeds: randomized + pushes)")):
    if prot not in A or prot not in B:
      continue
    a, b = A[prot], B[prot]
    lines += [f"\n### {title}\n", f"| metric | {la} | {lb} | b − a | paired test (b vs a) |", "|---|---|---|---|---|"]
    for k, lab, nd in ROWS:
      if k == "success" and prot == "clean":
        continue
      va, vb = a[k], b[k]
      test = ""
      if prot != "clean" and k in a.get("per_episode", {}) and k != "success":
        t = paired(a, b, k)
        test = f"median Δ {t['median_diff']:+.{max(nd, 2)}f}, b lower in {t['b_lower_frac']:.0%}, p = {t['p']:.2g}"
      f = lambda v: f"{v:.{nd}f}" if nd else f"{v:.0f}"
      lines.append(f"| {lab} | {f(va)} | {f(vb)} | {vb - va:+.{nd}f} | {test} |")
    if prot == "clean":
      ra, rb = a["repeat_body_mean_range"], b["repeat_body_mean_range"]
      lines.append(f"| body-mean local, 5 repeats (min–max) | {ra[0]:.2f}–{ra[1]:.2f} | {rb[0]:.2f}–{rb[1]:.2f} | | |")
  return "\n".join(lines)


def tb_scalars(train_dir: Path, tag):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  ev = sorted(train_dir.glob("events.out.tfevents*"))
  ea = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
  ea.Reload()
  s = ea.Scalars(tag)
  return np.array([x.step for x in s]), np.array([x.value for x in s])


def plot_lambda(train_b: Path, train_a: Path, out: Path):
  import csv
  rows = list(csv.DictReader(open(train_b / "leg_constraint.csv")))
  it = np.array([int(r["iteration"]) for r in rows])
  g = lambda k: np.array([float(r[k]) for r in rows])
  lam, ebar, ebat, eps, ebody = g("lambda"), g("e_bar"), g("e_batch"), g("eps")[0], g("e_body")
  lam_max = 2.0
  fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, facecolor=SURF,
                          gridspec_kw={"height_ratios": [1, 1.3, 1]})
  for ax in axs:
    style(ax)
  axs[0].plot(it, lam, color=COL["b"], lw=1.6)
  axs[0].axhline(lam_max, color=INK2, lw=0.8, ls="--")
  axs[0].text(it[0], lam_max, "  cap 2", va="bottom", fontsize=8, color=INK2)
  axs[0].set_ylim(0, lam_max * 1.12)
  axs[0].set_ylabel("lambda", color=INK2)
  axs[0].set_title("Multiplier lambda (after each iteration's update)", loc="left", fontsize=10, color=INK)
  axs[1].plot(it, 100 * ebat, color="#c3c2b7", lw=0.6, label="e_batch (rollout mean of e_leg)")
  axs[1].plot(it, 100 * ebar, color=COL["b"], lw=1.6, label="e_bar (EMA, 0.9)")
  axs[1].axhline(100 * eps, color=INK, lw=1.0, ls="--")
  axs[1].text(it[-1], 100 * eps, f"eps = {100 * eps:.3f} cm  ", ha="right", va="bottom", fontsize=8, color=INK)
  axs[1].set_ylabel("leg error [cm]", color=INK2)
  axs[1].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
  axs[1].set_title("Leg error during training (6 leg bodies, training sampling, stochastic actions)",
                   loc="left", fontsize=10, color=INK)
  sa, va = tb_scalars(train_a, "Metrics/motion/error_body_pos")
  roll = lambda v, w=100: np.convolve(v, np.ones(w) / w, mode="valid") if len(v) >= w else v[:0]
  for x, v, key, lab in ((sa, va, "a", "stock 10k→20k (Metrics/motion/error_body_pos)"),
                         (it, ebody, "b", "leg constraint (Constraint/e_body, same definition)")):
    axs[2].plot(x, 100 * v, color=COL[key], lw=0.4, alpha=0.25)
    r = roll(v)
    axs[2].plot(x[len(x) - len(r):], 100 * r, color=COL[key], lw=1.8, label=lab + ", 100-iteration mean")
  axs[2].set_ylabel("all-body error [cm]", color=INK2)
  axs[2].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
  axs[2].set_title("All-body local error during training, both runs (faint: per iteration)", loc="left",
                   fontsize=10, color=INK)
  lo = np.percentile(np.r_[va[50:], ebody[50:]], 0.5) * 100
  hi = np.percentile(np.r_[va[50:], ebody[50:]], 99.5) * 100
  axs[2].set_ylim(lo * 0.97, hi * 1.03)
  axs[2].set_xlabel("PPO iteration", color=INK2)
  fig.tight_layout()
  fig.savefig(out / "lambda_ebar.png", dpi=110, facecolor=SURF)
  plt.close(fig)
  return {"iterations": [int(it[0]), int(it[-1])], "lambda_final": float(lam[-1]), "lambda_max_seen": float(lam.max()),
          "iterations_at_cap": int((lam >= lam_max).sum()),
          "first_iteration_at_cap": int(it[np.argmax(lam >= lam_max)]) if (lam >= lam_max).any() else None,
          "e_bar_final_cm": float(100 * ebar[-1]), "e_batch_last100_cm": float(100 * ebat[-100:].mean()),
          "eps_cm": float(100 * eps), "e_body_last100_cm": float(100 * ebody[-100:].mean()),
          "stock_error_body_pos_last100_cm": float(100 * va[-100:].mean())}


def plot_sweep(runs, out: Path):
  fig, axs = plt.subplots(1, 3, figsize=(15, 4), facecolor=SURF, sharey=True)
  for ax, g in zip(axs, GROUPS):
    style(ax)
    for key, lab, sw in runs:
      sw = [s for s in sw if s["iteration"] > 0]
      x = [s["iteration"] for s in sw]
      y = [s["group_pos_local_cm"][g] for s in sw]
      ax.plot(x, y, color=COL[key], lw=1.4, marker="o", ms=3, label=lab)
    ax.set_title(f"{g}: clean-run local error per checkpoint", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("checkpoint iteration", color=INK2)
  axs[0].set_ylabel("pos error (local) [cm]", color=INK2)
  axs[0].legend(fontsize=8, frameon=False, labelcolor=INK)
  fig.tight_layout()
  fig.savefig(out / "group_error_vs_iteration.png", dpi=110, facecolor=SURF)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--a", type=Path, required=True, help="eval dir of the reference (stock) run")
  ap.add_argument("--b", type=Path, required=True, help="eval dir of the compared (constrained) run")
  ap.add_argument("--base", type=Path, help="eval dir of the shared 0-10k run (sweep plot only)")
  ap.add_argument("--la", default="stock")
  ap.add_argument("--lb", default="leg constraint")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--commit", default="")
  a = ap.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  A, B = summarize(a.a), summarize(a.b)
  train_a = Path(json.loads((a.a / "metrics.json").read_text())["checkpoint"]).parent
  train_b = Path(json.loads((a.b / "metrics.json").read_text())["checkpoint"]).parent
  lam = plot_lambda(train_b, train_a, a.out)
  runs = []
  if a.base:
    runs.append(("base", "shared 0–10k", json.loads((a.base / "metrics.json").read_text()).get("sweep", [])))
  runs += [("a", a.la, A["sweep"]), ("b", a.lb, B["sweep"])]
  plot_sweep(runs, a.out)
  md = [f"# {a.la} vs {a.lb}\n",
        f"- a = `{A['checkpoint']}` (task `{A['task']}`)", f"- b = `{B['checkpoint']}` (task `{B['task']}`)",
        f"- code: commit `{a.commit}`" if a.commit else "",
        f"- multiplier: {json.dumps(lam)}", table(A, B, a.la, a.lb)]
  (a.out / "comparison.md").write_text("\n".join(md) + "\n")
  strip = lambda R: {k: ({kk: vv for kk, vv in v.items() if kk != "per_episode"} if isinstance(v, dict) else v)
                     for k, v in R.items() if k != "sweep"}
  tests = {prot: {k: paired(A[prot], B[prot], k) for k, *_ in ROWS
                  if prot in A and prot in B and k in A[prot].get("per_episode", {})}
           for prot in ("randomized", "push")}
  (a.out / "comparison.json").write_text(json.dumps({"a": strip(A), "b": strip(B), "paired_tests": tests,
                                                    "multiplier": lam, "commit": a.commit}, indent=1))
  print("\n".join(md))
  print("->", a.out)


if __name__ == "__main__":
  main()
