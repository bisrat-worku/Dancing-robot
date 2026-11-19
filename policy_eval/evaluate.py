"""Evaluate a trained mjlab G1 tracking policy against its reference dance.

  python3 policy_eval/evaluate.py outputs/<run>/                 # everything (about 1 h on the RTX 5060)
  python3 policy_eval/evaluate.py outputs/<run>/ --no-second     # skip pushes / friction / DTW
  python3 policy_eval/evaluate.py outputs/<run>/ --rescore       # recompute metrics from saved rollouts

Writes outputs/<run>/eval/<training-run>/ :
  metrics.json  episodes.csv  report.md  clean_run.mp4  plots/*.png  raw/*.npz (trajectories)  logs/
Re-launches itself in mjlab's venv; music beats run in EDGE's venv (librosa).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata as im
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MJLAB_DIR = Path(os.environ.get("MJLAB_DIR", "/home/bwork/mjlab"))
MJLAB_PY = MJLAB_DIR / ".venv" / "bin" / "python"
try:
  import mjlab  # noqa: F401
except ImportError:
  if Path(sys.executable).resolve() == MJLAB_PY.resolve():
    raise
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
from config import EvalConfig, task_for  # noqa: E402

EDGE_ROOT = HERE.parent
EDGE_PY = EDGE_ROOT / ".venv" / "bin" / "python"
FFMPEG = EDGE_ROOT / ".venv" / "bin" / "ffmpeg"


# =============================================================================== helpers


def jdefault(o):
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return None if not np.isfinite(o) else float(o)
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, Path):
    return str(o)
  raise TypeError(type(o))


def clean_json(x):
  """NaN/inf -> None so metrics.json is strict JSON."""
  if isinstance(x, dict):
    return {k: clean_json(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [clean_json(v) for v in x]
  if isinstance(x, float) and not np.isfinite(x):
    return None
  return x


def sha256(p: Path):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
      h.update(b)
  return h.hexdigest()


def run_cmd(cmd, log: Path):
  log.parent.mkdir(parents=True, exist_ok=True)
  print("  $", " ".join(str(c) for c in cmd), flush=True)
  with open(log, "w") as f:
    p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=EDGE_ROOT)
  if p.returncode != 0:
    tail = log.read_text().splitlines()[-30:]
    raise SystemExit(f"command failed ({p.returncode}), see {log}:\n" + "\n".join(tail))
  return log.read_text()


def load_npz(path):
  d = np.load(path)
  meta = {k: d[k] for k in d.files if "/" not in k}
  eps = []
  for i in range(int(meta["n_episodes"])):
    p = f"ep{i}/"
    eps.append({k[len(p):]: d[k] for k in d.files if k.startswith(p)})
  return meta, eps


def git(repo, *args):
  try:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()
  except Exception:
    return None


def as_motion(src, fps, n=None, prefix=""):
  keys = ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel", "joint_pos", "joint_vel")
  m = {k: np.asarray(src[prefix + k], dtype=np.float64)[:n] for k in keys}
  m["fps"] = fps
  return m


def load_reference(npz):
  r = np.load(npz)
  m = {"body_pos": r["body_pos_w"], "body_quat": r["body_quat_w"], "body_lin_vel": r["body_lin_vel_w"],
       "body_ang_vel": r["body_ang_vel_w"], "joint_pos": r["joint_pos"], "joint_vel": r["joint_vel"]}
  m = {k: np.asarray(v, dtype=np.float64) for k, v in m.items()}
  m["fps"] = float(r["fps"][0])
  return m


def slice_motion(m, a, b):
  out = {k: v[a:b] for k, v in m.items() if isinstance(v, np.ndarray)}
  out["fps"] = m["fps"]
  return out


# =============================================================================== scoring


class Ctx:
  """Everything that is the same for every episode."""

  def __init__(self, ref, meta, cfg: EvalConfig, beats):
    self.ref, self.meta, self.cfg, self.beats = ref, meta, cfg, beats
    self.fps = ref["fps"]
    self.T = len(ref["body_pos"])
    self.body_names = [str(x) for x in meta["body_names"]]
    self.joint_names = [str(x) for x in meta["joint_names"]]
    self.bi = {n: i for i, n in enumerate(self.body_names)}
    self.anchor = self.bi["torso_link"]
    self.tracked = None  # set from the first episode (the task's motion body_names)
    self.foot = {s: self.bi[n] for s, n in cfg.feet.items()}
    self.toe = {"left": meta["toe_left"], "right": meta["toe_right"]}
    self.heel = {"left": meta["heel_left"], "right": meta["heel_right"]}
    self.sigma_frames = cfg.beat_sigma_s * self.fps
    self.max_lag = int(round(cfg.max_lag_s * self.fps))
    act_joint = meta["actuator_joint"]
    self.act_of_joint = np.argsort(act_joint)  # joint j <- actuator act_of_joint[j]
    assert (act_joint[self.act_of_joint] == np.arange(len(self.joint_names))).all()
    self.force_lim = np.abs(meta["actuator_forcerange"]).max(1)[self.act_of_joint]
    self.jrange = meta["joint_range"]
    self.vlim = meta["joint_vel_limit"]


def motion_only(ctx: Ctx, m, music_beats=None):
  """Metrics defined on a motion alone (computed identically for reference and executed)."""
  cfg = ctx.cfg
  n = len(m["body_pos"])
  out = {"n_frames": n}
  bp = m["body_pos"]
  # PFC
  if n >= 4:
    toe = {s: M.body_point(m, ctx.foot[s], ctx.toe[s]) for s in ("left", "right")}
    out["pfc"] = M.pfc(bp[:, ctx.bi["pelvis"]], bp[:, ctx.foot["left"]], toe["left"],
                       bp[:, ctx.foot["right"]], toe["right"], ctx.fps)
  else:
    out["pfc"] = None
  # beats
  kb, kvs = M.kinematic_beats(bp, ctx.fps, cfg.kinetic_smooth_sigma_s)
  out["kinematic_beats"] = kb
  mb = np.asarray(music_beats if music_beats is not None else ctx.beats["song"]["beat_frames"])
  mb = mb[mb < n - 1]
  out["bas"] = M.beat_alignment(kb, mb, ctx.sigma_frames)
  out["bas"]["chance_kin_to_music"] = M.beat_alignment_chance(kb, mb, ctx.sigma_frames)
  out["bas_calibration"] = {}
  for spec, cal in ctx.beats["calibration"].items():
    cb = np.asarray(cal["beat_frames"])
    out["bas_calibration"][spec] = M.beat_alignment(kb, cb[cb < n - 1], ctx.sigma_frames)
  # smoothness / limits
  jerk = M.joint_jerk(m["joint_pos"], ctx.fps)
  out["joint_jerk_mean"] = float(np.mean(jerk)) if jerk.size else None
  out["joint_jerk_p95"] = float(np.percentile(jerk, 95)) if jerk.size else None
  out["joint_jerk_per_joint_mean"] = dict(zip(ctx.joint_names, jerk.mean(0))) if jerk.size else {}
  marg = np.radians(cfg.joint_limit_margin_deg)
  near = M.near_limit_fraction(m["joint_pos"], ctx.jrange[:, 0], ctx.jrange[:, 1], marg)
  out["near_pos_limit_frac_per_joint"] = dict(zip(ctx.joint_names, near))
  q = m["joint_pos"]
  near_any = ((q <= ctx.jrange[:, 0] + marg) | (q >= ctx.jrange[:, 1] - marg)).any(1)
  out["near_pos_limit_frac_any_joint"] = float(near_any.mean())
  out["near_pos_limit_frac_mean_joint"] = float(near.mean())
  out["beyond_pos_limit_frac_any_joint"] = float(((q < ctx.jrange[:, 0]) | (q > ctx.jrange[:, 1])).any(1).mean())
  qd = np.abs(m["joint_vel"])
  nv = qd >= (1 - cfg.velocity_limit_margin_frac) * ctx.vlim
  out["near_vel_limit_frac_per_joint"] = dict(zip(ctx.joint_names, nv.mean(0)))
  out["near_vel_limit_frac_any_joint"] = float(nv.any(1).mean())
  out["near_vel_limit_frac_mean_joint"] = float(nv.mean())
  out["beyond_vel_limit_frac_any_joint"] = float((qd > ctx.vlim).any(1).mean())
  out["peak_joint_speed"] = float(qd.max())
  out["peak_joint_speed_joint"] = ctx.joint_names[int(np.argmax(qd.max(0)))]
  return out


def point_velocity(m, b, offset):
  """Exact rigid-body velocity of a point fixed in body b: v + w x (R offset)."""
  q = m["body_quat"][:, b]
  r = M.quat_apply(q, np.broadcast_to(offset, q.shape[:-1] + (3,)))
  return m["body_lin_vel"][:, b] + np.cross(m["body_ang_vel"][:, b], r)


def score_episode(ctx: Ctx, ep, want_dtw=False):
  cfg = ctx.cfg
  n = len(ep["body_pos"])
  if ctx.tracked is None:
    ctx.tracked = [str(x) for x in ep["tracked_body_names"]]
  tr = [ctx.bi[b] for b in ctx.tracked]
  exe = as_motion(ep, ctx.fps)
  ref = slice_motion(ctx.ref, 0, n)
  te = M.tracking_errors(ref, exe, tr, ctx.anchor)
  s = slice(1, n)  # frame 0 is the reset state, not something the policy produced
  first = int(ep["first_term_frame"])
  completed = first < 0 and n - 1 == ctx.T - 1
  summ = {
    "seed": int(ep["seed"]), "checkpoint": str(ep["checkpoint"]),
    "completed": bool(completed),
    "first_term_frame": first if first >= 0 else None,
    "first_term_time_s": first / ctx.fps if first >= 0 else None,
    "first_term_conditions": [str(x) for x in ep["fired"]],
    "completion_fraction": (n - 1) / (ctx.T - 1),
    "frames_scored": n - 1,
  }
  bm = te["pos_local_cm"][s].mean(1)
  summ["pos_local_body_mean_max_cm"] = float(bm.max()) if bm.size else None
  for X in cfg.pos_error_thresholds_cm:
    summ[f"under_{X:g}cm"] = bool(completed and bm.size and bm.max() <= X)
  for k in ("pos_local_cm", "pos_global_cm", "ori_local_deg", "ori_global_deg", "linvel_global",
            "linvel_local", "acc_global"):
    summ[f"{k}_mean"] = float(te[k][s].mean()) if n > 1 else None
  summ["joint_abs_deg_mean"] = float(te["joint_abs_deg"][s].mean()) if n > 1 else None
  summ["joint_l2_rad_mean"] = float(te["joint_l2_rad"][s].mean()) if n > 1 else None
  # motion-only metrics on executed and on the same reference segment
  summ["exec"] = motion_only(ctx, exe)
  summ["ref_same_segment"] = motion_only(ctx, ref)
  # lag
  if n >= 4 * ctx.max_lag:
    L, r, sub = M.lag_frames(M.kinetic_velocity(ref["body_pos"]), M.kinetic_velocity(exe["body_pos"]),
                             ctx.max_lag)
    summ["lag_frames"], summ["lag_ms"], summ["lag_corr"] = L, 1000 * L / ctx.fps, r
    summ["lag_ms_subframe"] = 1000 * sub / ctx.fps
  else:
    summ["lag_frames"] = summ["lag_ms"] = summ["lag_corr"] = summ["lag_ms_subframe"] = None
  # foot slip (executed, simulator contact)
  slip = {}
  for i, side in enumerate(("left", "right")):
    c = ep["contact_found"][s, i] > 0.5
    b = ctx.foot[side]
    v_org = 100 * np.linalg.norm(exe["body_lin_vel"][s, b, :2], axis=-1)
    v_toe = 100 * np.linalg.norm(point_velocity(exe, b, ctx.toe[side])[s, :2], axis=-1)
    v_heel = 100 * np.linalg.norm(point_velocity(exe, b, ctx.heel[side])[s, :2], axis=-1)
    v_min = np.minimum(v_toe, v_heel)
    slip[side] = {"contact_frac": float(c.mean()) if c.size else None,
                  "ankle_speed_cm_s": M.stats(v_org[c]),
                  "sole_min_speed_cm_s": M.stats(v_min[c]),
                  "frac_contact_ankle_gt_thresh": float((v_org[c] > cfg.slip_speed_threshold_cm_s).mean()) if c.any() else None,
                  "frac_contact_sole_min_gt_thresh": float((v_min[c] > cfg.slip_speed_threshold_cm_s).mean()) if c.any() else None}
  summ["foot_slip"] = slip
  # effort (executed)
  tq = ep["torque"][s][:, :, ctx.act_of_joint]  # (n-1, decimation, J) joint order
  atq = np.abs(tq)
  summ["torque_mean_abs_per_joint"] = dict(zip(ctx.joint_names, atq.mean((0, 1)))) if n > 1 else {}
  summ["torque_peak_abs_per_joint"] = dict(zip(ctx.joint_names, atq.max((0, 1)))) if n > 1 else {}
  sat = atq >= cfg.torque_limit_frac * ctx.force_lim
  summ["torque_sat_frac_per_joint"] = dict(zip(ctx.joint_names, sat.mean((0, 1)))) if n > 1 else {}
  summ["torque_sat_frac_any_joint"] = float(sat.any(2).mean()) if n > 1 else None
  summ["torque_sat_frac_mean_joint"] = float(sat.mean()) if n > 1 else None
  summ["torque_mean_abs"] = float(atq.mean()) if n > 1 else None
  summ["torque_peak_abs"] = float(atq.max()) if n > 1 else None
  summ["torque_peak_ratio"] = float((atq.max((0, 1)) / ctx.force_lim).max()) if n > 1 else None
  a = ep["action"][s]
  if len(a) >= 2:
    rate = np.sum((a[1:] - a[:-1]) ** 2, axis=-1)
    summ["action_rate_l2"] = M.stats(rate)
  else:
    summ["action_rate_l2"] = M.stats([])
  if want_dtw and n > 2:
    band = ctx.max_lag
    summ["dtw_pos_global_cm"], summ["dtw_path_len"] = M.dtw_position_error(
      ref["body_pos"][s][:, tr], exe["body_pos"][s][:, tr], band)
    summ["dtw_band_frames"] = band
  return summ, te


def pooled_tracking(ctx: Ctx, tes):
  """Stats over all (episode, frame) samples, frames 1.. of each episode, per body and body mean."""
  out = {}
  for k in ("pos_local_cm", "pos_global_cm", "ori_local_deg", "ori_global_deg", "linvel_global",
            "linvel_local", "acc_global"):
    arr = np.concatenate([t[k][1:] for t in tes], 0)
    out[k] = {"body_mean": M.stats(arr.mean(1)),
              "per_body": {b: M.stats(arr[:, i]) for i, b in enumerate(ctx.tracked)}}
  arr = np.concatenate([t["joint_abs_deg"][1:] for t in tes], 0)
  out["joint_abs_deg"] = {"joint_mean": M.stats(arr.mean(1)),
                          "per_joint": {j: M.stats(arr[:, i]) for i, j in enumerate(ctx.joint_names)}}
  out["joint_l2_rad"] = M.stats(np.concatenate([t["joint_l2_rad"][1:] for t in tes]))
  return out


def episode_set_summary(ctx, summs):
  n = len(summs)
  ok = [s["completed"] for s in summs]
  out = {"n_episodes": n, "success_rate": float(np.mean(ok)) if n else None,
         "n_success": int(np.sum(ok)),
         "completion_fraction": M.stats([s["completion_fraction"] for s in summs]),
         "first_term_times_s": [s["first_term_time_s"] for s in summs],
         "first_term_conditions": {}}
  for s in summs:
    for c in s["first_term_conditions"]:
      out["first_term_conditions"][c] = out["first_term_conditions"].get(c, 0) + 1
  for X in ctx.cfg.pos_error_thresholds_cm:
    out[f"frac_under_{X:g}cm"] = float(np.mean([s[f"under_{X:g}cm"] for s in summs])) if n else None
  for k in ("lag_ms", "lag_ms_subframe", "torque_sat_frac_any_joint", "torque_mean_abs", "torque_peak_abs"):
    vals = [s[k] for s in summs if s[k] is not None]
    out[k] = M.stats(vals)
  out["pfc_exec"] = M.stats([s["exec"]["pfc"] for s in summs if s["exec"]["pfc"] is not None])
  out["bas_exec_kin_to_music"] = M.stats([s["exec"]["bas"]["kin_to_music"] for s in summs])
  out["seeds"] = [s["seed"] for s in summs]
  return out


# =============================================================================== self-tests


def self_tests(ctx: Ctx, clean_ep):
  res = []
  ref = ctx.ref
  tr = [ctx.bi[b] for b in ctx.tracked]

  def add(name, passed, detail):
    res.append({"test": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

  # 1. reference vs itself
  te = M.tracking_errors(ref, ref, tr, ctx.anchor)
  mx = {k: float(np.max(v)) for k, v in te.items()}
  kv = M.kinetic_velocity(ref["body_pos"])
  L, r, sub = M.lag_frames(kv, kv, ctx.max_lag)
  add("reference vs itself: zero error",
      all(v < 1e-6 for k, v in mx.items() if k not in ("ori_local_deg", "ori_global_deg"))
      and mx["ori_local_deg"] < 1e-3 and mx["ori_global_deg"] < 1e-3,
      "max over frames/bodies: " + ", ".join(f"{k}={v:.2e}" for k, v in mx.items()))
  add("reference vs itself: zero lag", L == 0 and abs(sub) < 1e-6,
      f"lag={L} frames (sub-frame {sub:.3g}), corr={r:.6f}")
  # 2. shifted copies
  for shift in (7, -4, 1):
    sh = {k: np.roll(v, shift, 0) for k, v in ref.items() if isinstance(v, np.ndarray)}
    for k in sh:
      if shift > 0:
        sh[k][:shift] = ref[k][0]
      else:
        sh[k][shift:] = ref[k][-1]
    sh["fps"] = ref["fps"]
    L, r, sub = M.lag_frames(kv, M.kinetic_velocity(sh["body_pos"]), ctx.max_lag)
    add(f"reference shifted by {shift:+d} frames: lag = {shift:+d}", L == shift,
        f"lag={L} frames = {1000 * L / ctx.fps:.0f} ms (sub-frame {sub:.2f}), corr={r:.4f}")
  # 3. wrong song
  right = M.beat_alignment(M.kinematic_beats(ref["body_pos"], ctx.fps, ctx.cfg.kinetic_smooth_sigma_s)[0],
                           np.asarray(ctx.beats["song"]["beat_frames"]), ctx.sigma_frames)
  kb = M.kinematic_beats(ref["body_pos"], ctx.fps, ctx.cfg.kinetic_smooth_sigma_s)[0]
  for spec, cal in ctx.beats["calibration"].items():
    wrong = M.beat_alignment(kb, np.asarray(cal["beat_frames"]), ctx.sigma_frames)
    add(f"wrong song ({spec}) scores lower than the right song (reference motion)",
        wrong["kin_to_music"] < right["kin_to_music"],
        f"kin->music right {right['kin_to_music']:.4f} vs wrong {wrong['kin_to_music']:.4f}; "
        f"music->kin right {right['music_to_kin']:.4f} vs wrong {wrong['music_to_kin']:.4f}")
  # 3b. positive control: a synthetic motion whose speed minima sit on the music beats
  mb_song = np.asarray(ctx.beats["song"]["beat_frames"])
  syn = M.synthetic_beat_motion(mb_song, ctx.T)
  kbs = M.kinematic_beats(syn, ctx.fps, ctx.cfg.kinetic_smooth_sigma_s)[0]
  r_syn = M.beat_alignment(kbs, mb_song, ctx.sigma_frames)
  w_syn = {spec: M.beat_alignment(kbs, np.asarray(cal["beat_frames"]), ctx.sigma_frames)
           for spec, cal in ctx.beats["calibration"].items()}
  ch_syn = M.beat_alignment_chance(kbs, mb_song, ctx.sigma_frames)
  add("positive control: synthetic motion with speed minima on the song's beats scores above its chance "
      "mean on the right song and lower on every wrong song",
      r_syn["kin_to_music"] > ch_syn["mean"] and all(w["kin_to_music"] < r_syn["kin_to_music"] for w in w_syn.values()),
      f"right {r_syn['kin_to_music']:.4f} (chance mean {ch_syn['mean']:.4f}; {r_syn['n_kinematic_beats']} kinematic "
      f"beats for {len(mb_song)} music beats, the extra ones lie before the first / after the last music beat); wrong: "
      + ", ".join(f"{k} {v['kin_to_music']:.4f}" for k, v in w_syn.items()))
  ch = M.beat_alignment_chance(kb, mb_song, ctx.sigma_frames)
  res[-1]["detail"] += (f". For the reference motion itself: right-song score {right['kin_to_music']:.4f} vs "
                        f"chance {ch['mean']:.4f} (p5 {ch['p5']:.4f}, p95 {ch['p95']:.4f})")
  # 4. corrupted limb
  limb = [b for b in ctx.body_names if b.startswith("left_shoulder") or b.startswith("left_elbow")
          or b.startswith("left_wrist")]
  bad = {k: v.copy() for k, v in ref.items() if isinstance(v, np.ndarray)}
  bad["fps"] = ref["fps"]
  for b in limb:
    bad["body_pos"][:, ctx.bi[b]] += np.array([0.10, 0.0, 0.0])
  te = M.tracking_errors(ref, bad, tr, ctx.anchor)
  per = {b: (float(te["pos_local_cm"][:, i].mean()), float(te["pos_global_cm"][:, i].mean()))
         for i, b in enumerate(ctx.tracked)}
  hit = [b for b in ctx.tracked if b in limb]
  ok = all(abs(per[b][0] - 10) < 1e-3 and abs(per[b][1] - 10) < 1e-3 for b in hit) and \
    all(per[b][0] < 1e-6 and per[b][1] < 1e-6 for b in ctx.tracked if b not in limb)
  add("left arm offset by 10 cm shows up only in the left-arm bodies", ok,
      "; ".join(f"{b}: local {per[b][0]:.3f} / global {per[b][1]:.3f} cm" for b in ctx.tracked))
  # 5. numpy anchor-relative frame == mjlab's (recorded from the env during the clean run)
  if clean_ep is not None and "mj_rel_pos" in clean_ep:
    exe = as_motion(clean_ep, ctx.fps)
    n = len(exe["body_pos"])
    rf = clean_ep["rel_frame"][1:]
    refk = {k: ref[k][rf] for k in ("body_pos", "body_quat")}
    exk = {k: exe[k][1:n] for k in ("body_pos", "body_quat")}
    pos, quat, _ = M.anchor_relative_reference(refk, exk, ctx.anchor)
    dp = np.abs(pos[:, tr] - clean_ep["mj_rel_pos"][1:]).max()
    dq = np.degrees(M.quat_angle(quat[:, tr], clean_ep["mj_rel_quat"][1:].astype(np.float64))).max()
    add("numpy anchor-relative reference equals mjlab's body_pos_relative_w / body_quat_relative_w",
        dp < 1e-4 and dq < 0.05, f"max |dpos| = {dp * 1000:.4f} mm, max angle = {dq:.4f} deg over {n - 1} frames")
  return res


# =============================================================================== training logs


def training_info(run: Path, train_dir: Path):
  info = {"train_dir": str(train_dir)}
  agent = yaml.load((train_dir / "params" / "agent.yaml").read_text(), Loader=yaml.BaseLoader)
  env = yaml.load((train_dir / "params" / "env.yaml").read_text(), Loader=yaml.BaseLoader)
  info["seed"] = int(agent["seed"])
  info["num_envs"] = int(env["scene"]["num_envs"])
  info["num_steps_per_env"] = int(agent["num_steps_per_env"])
  info["max_iterations"] = int(agent["max_iterations"])
  info["episode_length_s"] = float(env["episode_length_s"])
  info["logger"] = agent["logger"]
  ev = sorted(train_dir.glob("events.out.tfevents*"))
  if ev:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    info["logged_scalars"] = tags
    info["per_body_scalars"] = [t for t in tags if any(b in t for b in ("_link", "pelvis"))]
    s = ea.Scalars("Perf/total_fps")
    info["iterations_logged"] = len(s)
    wall = [x.wall_time for x in s]
    info["wall_clock_s"] = wall[-1] - wall[0]
    info["wall_clock_h"] = info["wall_clock_s"] / 3600
    info["started"] = dt.datetime.fromtimestamp(wall[0]).isoformat(timespec="seconds")
    info["finished"] = dt.datetime.fromtimestamp(wall[-1]).isoformat(timespec="seconds")
    fps = np.array([x.value for x in s])
    info["env_steps_per_s"] = M.stats(fps)
    info["env_steps_total"] = len(s) * info["num_steps_per_env"] * info["num_envs"]
    info["env_steps_per_s_overall"] = info["env_steps_total"] / max(info["wall_clock_s"], 1e-9)
    for k in ("Perf/collection_time", "Perf/learning_time"):
      info[k.split("/")[1] + "_s_per_iter"] = M.stats([x.value for x in ea.Scalars(k)])
    info["final_logged"] = {t: ea.Scalars(t)[-1].value for t in tags
                            if t.startswith(("Metrics/motion/error", "Train/", "Episode_Termination/"))}
  return info


def all_training_runs(run: Path):
  rows = []
  for d in sorted((run / "rl" / "g1_tracking").glob("*/")):
    ck = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    agent = yaml.load((d / "params" / "agent.yaml").read_text(), Loader=yaml.BaseLoader) \
      if (d / "params" / "agent.yaml").exists() else {}
    mi = int(agent.get("max_iterations", -1))
    last = int(ck[-1].stem.split("_")[1]) if ck else None
    envy = (d / "params" / "env.yaml").read_text() if (d / "params" / "env.yaml").exists() else ""
    ne = next((int(l.split(":")[1]) for l in envy.splitlines() if l.startswith("  num_envs:")), -1)
    rows.append({"dir": d.name, "seed": int(agent.get("seed", -1)), "max_iterations": mi, "num_envs": ne,
                 "checkpoints": [int(c.stem.split("_")[1]) for c in ck], "last_iteration": last,
                 "complete": last is not None and last >= mi - 1,
                 "task": task_for(d) if (d / "params" / "env.yaml").exists() else None})
  return rows


# =============================================================================== plots


def plot_all(out, ctx, clean_summ, clean_te, rand_summs, rand_tes, sweep, open_summ, push_summs):
  pd = out / "plots"
  pd.mkdir(exist_ok=True)
  fps = ctx.fps
  # 1. error over time (clean)
  t = np.arange(len(clean_te["pos_local_cm"])) / fps
  fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
  ax[0].plot(t, clean_te["pos_local_cm"].mean(1), label="local (reward frame)")
  ax[0].plot(t, clean_te["pos_global_cm"].mean(1), label="global (world)")
  ax[0].set_ylabel("body-mean pos err [cm]")
  ax[0].legend(loc="upper left")
  ax[1].plot(t, clean_te["ori_local_deg"].mean(1), label="local")
  ax[1].plot(t, clean_te["ori_global_deg"].mean(1), label="global")
  ax[1].set_ylabel("body-mean ori err [deg]")
  ax[1].legend(loc="upper left")
  ax[2].plot(t, clean_te["linvel_global"].mean(1))
  ax[2].set_ylabel("body-mean lin vel err [m/s]")
  ax[3].plot(t, clean_te["joint_abs_deg"].mean(1))
  ax[3].set_ylabel("joint-mean |dq| [deg]")
  ax[3].set_xlabel(f"time [s] (0 = {ctx.beats['song']['offset_s']} s into the song)")
  for a in ax:
    a.grid(alpha=0.3)
    for b in ctx.beats["song"]["beat_frames"]:
      a.axvline(b / fps, color="k", alpha=0.06, lw=0.8)
  if clean_summ["first_term_frame"] is not None:
    for a in ax:
      a.axvline(clean_summ["first_term_time_s"], color="r", ls="--")
  fig.suptitle("Clean run: tracking error over time (grey lines = music beats)")
  fig.tight_layout()
  fig.savefig(pd / "error_over_time.png", dpi=110)
  plt.close(fig)
  # 2. per-body error (clean + randomized pooled)
  names = ctx.tracked
  x = np.arange(len(names))
  fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
  for i, (key, lab) in enumerate((("pos_local_cm", "local pos err [cm]"), ("pos_global_cm", "global pos err [cm]"))):
    c = clean_te[key][1:]
    r = np.concatenate([tt[key][1:] for tt in rand_tes], 0) if rand_tes else None
    ax[i].bar(x - 0.2, c.mean(0), 0.4, label="clean: mean")
    ax[i].scatter(x - 0.2, np.percentile(c, 95, axis=0), marker="_", s=200, c="C0", label="clean: p95")
    if r is not None:
      ax[i].bar(x + 0.2, r.mean(0), 0.4, label=f"randomized ({len(rand_tes)} ep): mean")
      ax[i].scatter(x + 0.2, np.percentile(r, 95, axis=0), marker="_", s=200, c="C1", label="randomized: p95")
    ax[i].set_ylabel(lab)
    ax[i].grid(alpha=0.3, axis="y")
    ax[i].legend(fontsize=8)
  ax[1].set_xticks(x, [n.replace("_link", "") for n in names], rotation=40, ha="right")
  fig.suptitle("Per-body position error (bars = mean, ticks = 95th percentile)")
  fig.tight_layout()
  fig.savefig(pd / "per_body_error.png", dpi=110)
  plt.close(fig)
  # 3. per-body error vs iteration
  if sweep:
    its = np.array([s["iteration"] for s in sweep])
    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for gi, (g, bodies) in enumerate(ctx.cfg.body_groups.items()):
      for b in bodies:
        y = np.array([s["per_body_pos_local_cm"][b] for s in sweep])
        ln, = ax[gi].plot(its, y, "-", label=b.replace("_link", ""))
        done = np.array([s["completed"] for s in sweep])
        ax[gi].scatter(its[done], y[done], c=ln.get_color(), s=14)
        ax[gi].scatter(its[~done], y[~done], facecolors="none", edgecolors=ln.get_color(), s=30)
      ax[gi].set_ylabel(f"{g}: mean local pos err [cm]")
      ax[gi].set_yscale("log")
      ax[gi].grid(alpha=0.3, which="both")
      ax[gi].legend(fontsize=7, ncol=3)
    ax[3].plot(its, [s["completion_fraction"] for s in sweep], "k-o", ms=4)
    ax[3].set_ylabel("clean-run completion")
    ax[3].set_xlabel("training iteration")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].grid(alpha=0.3)
    fig.suptitle("Clean run per checkpoint (filled = completed the clip, hollow = terminated early;\n"
                 "errors are averaged over the frames before termination)")
    fig.tight_layout()
    fig.savefig(pd / "per_body_error_vs_iteration.png", dpi=110)
    plt.close(fig)
  # 4. first termination times
  fig, ax = plt.subplots(figsize=(10, 4.5))
  clip = (ctx.T - 1) / fps
  rows = [("randomized", rand_summs), ("randomized + pushes", push_summs or []),
          ("open-loop", [open_summ] if open_summ else [])]
  for i, (lab, ss) in enumerate(rows):
    if not ss:
      continue
    tt = np.array([s["first_term_time_s"] if s["first_term_time_s"] is not None else clip for s in ss])
    done = np.array([s["completed"] for s in ss])
    jit = (np.random.default_rng(0).random(len(tt)) - 0.5) * 0.3
    ax.scatter(tt[~done], i + jit[~done], c="C3", s=18, label="terminated" if i == 0 else None)
    ax.scatter(tt[done], i + jit[done], c="C2", s=18, marker="s", label="completed (plotted at clip end)" if i == 0 else None)
  ax.axvline(clip, color="k", ls="--", lw=1)
  ax.set_yticks(range(len(rows)), [f"{r[0]} (n={len(r[1])})" for r in rows])
  ax.set_xlabel("first-termination time [s]")
  ax.set_xlim(-0.5, clip + 1)
  ax.legend(loc="lower left", fontsize=8)
  ax.grid(alpha=0.3)
  ax.set_title("First-termination times (dashed = end of the clip)")
  fig.tight_layout()
  fig.savefig(pd / "first_termination_times.png", dpi=110)
  plt.close(fig)
  # 5. kinetic velocity and beats (reference vs executed, clean)
  n = clean_summ["frames_scored"] + 1
  fig, ax = plt.subplots(figsize=(12, 4))
  _, kr = M.kinematic_beats(ctx.ref["body_pos"][:n], fps, ctx.cfg.kinetic_smooth_sigma_s)
  ax.plot(np.arange(len(kr)) / fps, kr * fps, label="reference (smoothed)")
  ax.plot(np.arange(len(kr)) / fps, M.kinematic_beats(clean_te["_exe_body_pos"], fps,
          ctx.cfg.kinetic_smooth_sigma_s)[1] * fps, label="executed (smoothed)", alpha=0.8)
  for b in ctx.beats["song"]["beat_frames"]:
    ax.axvline(b / fps, color="k", alpha=0.15, lw=0.8)
  ax.set_xlabel("time [s]")
  ax.set_ylabel("kinetic velocity [m/s]")
  ax.legend()
  ax.set_title("Kinetic velocity (mean body speed); grey = librosa music beats")
  fig.tight_layout()
  fig.savefig(pd / "kinetic_velocity_beats.png", dpi=110)
  plt.close(fig)


# =============================================================================== report


def fmt(x, nd=2):
  if x is None:
    return "n/a"
  if isinstance(x, (bool, np.bool_)):
    return "yes" if x else "no"
  if isinstance(x, (int, np.integer)):
    return str(int(x))
  return f"{x:.{nd}f}"


def write_report(out, R):
  cfg = R["config"]
  c, cr = R["clean"]["summary"], R["clean"]["summary"]["ref_same_segment"]
  ce = c["exec"]
  rs = R["randomized"]["set"]
  L = []
  a = L.append
  a(f"# G1 tracking policy evaluation: {R['song']}\n")
  a(f"Training run `{Path(R['training']['train_dir']).name}`, checkpoint `{Path(R['checkpoint']).name}`, "
    f"task `{cfg['task']}`. Generated {R['generated']} by `policy_eval/evaluate.py`.\n")
  a("## Reference vs executed motion\n")
  a("Executed = the clean run (deterministic policy, no randomization). For the randomized column, each "
    "motion-only metric is computed per episode and then averaged over the episodes; for an episode that "
    "terminated early, the metric covers the frames before the termination.\n")
  a("| metric | reference | executed (clean) | difference | executed (randomized, mean over episodes) |")
  a("|---|---|---|---|---|")

  def row(name, rv, ev, rand=None, nd=3):
    d = None if (rv is None or ev is None) else ev - rv
    a(f"| {name} | {fmt(rv, nd)} | {fmt(ev, nd)} | {fmt(d, nd)} | {fmt(rand, nd)} |")

  rmean = lambda f: (float(np.mean([f(s) for s in R["randomized"]["episodes"] if f(s) is not None]))
                     if R["randomized"]["episodes"] else None)
  row("PFC (EDGE formula, G1 bodies, 50 fps)", cr["pfc"], ce["pfc"], rmean(lambda s: s["exec"]["pfc"]))
  row("Beat alignment, kinematic->music (primary)", cr["bas"]["kin_to_music"], ce["bas"]["kin_to_music"],
      rmean(lambda s: s["exec"]["bas"]["kin_to_music"]))
  row("Beat alignment, music->kinematic (Bailando/EDGE)", cr["bas"]["music_to_kin"], ce["bas"]["music_to_kin"],
      rmean(lambda s: s["exec"]["bas"]["music_to_kin"]))
  row("  chance level of that score (random phase of the beat grid)", cr["bas"]["chance_kin_to_music"]["mean"],
      ce["bas"]["chance_kin_to_music"]["mean"])
  row("number of kinematic beats", cr["bas"]["n_kinematic_beats"], ce["bas"]["n_kinematic_beats"])
  row("joint jerk, mean [rad/s^3]", cr["joint_jerk_mean"], ce["joint_jerk_mean"],
      rmean(lambda s: s["exec"]["joint_jerk_mean"]), nd=0)
  row("joint jerk, p95 [rad/s^3]", cr["joint_jerk_p95"], ce["joint_jerk_p95"], nd=0)
  row("frames with any joint near a position limit", cr["near_pos_limit_frac_any_joint"],
      ce["near_pos_limit_frac_any_joint"], rmean(lambda s: s["exec"]["near_pos_limit_frac_any_joint"]))
  row("joint-frames near a position limit (mean over joints)", cr["near_pos_limit_frac_mean_joint"],
      ce["near_pos_limit_frac_mean_joint"])
  row("frames with any joint beyond its position range", cr["beyond_pos_limit_frac_any_joint"],
      ce["beyond_pos_limit_frac_any_joint"])
  row("frames with any joint near a velocity limit", cr["near_vel_limit_frac_any_joint"],
      ce["near_vel_limit_frac_any_joint"], rmean(lambda s: s["exec"]["near_vel_limit_frac_any_joint"]))
  row("frames with any joint beyond its velocity limit", cr["beyond_vel_limit_frac_any_joint"],
      ce["beyond_vel_limit_frac_any_joint"])
  row("peak joint speed [rad/s]", cr["peak_joint_speed"], ce["peak_joint_speed"], nd=1)
  a(f"| lag of executed behind reference [ms] | 0 (by definition) | {fmt(c['lag_ms'], 0)} "
    f"(sub-frame {fmt(c['lag_ms_subframe'], 1)}, r = {fmt(c['lag_corr'], 3)}) | | "
    f"{fmt(R['randomized']['set']['lag_ms']['mean'], 1)} |")
  ref_slip = R.get("reference_foot_slip")
  for side in ("left", "right"):
    sl = c["foot_slip"][side]
    rtxt = "n/a (see note)"
    if ref_slip:
      rtxt = (f"heel {fmt(ref_slip[side + '_heel']['g1_speed_cm_s']['mean'], 1)} / toe "
              f"{fmt(ref_slip[side + '_toe']['g1_speed_cm_s']['mean'], 1)} (EDGE contact labels, 30 fps)")
    a(f"| {side} foot slip while in contact, mean [cm/s] | {rtxt} | ankle "
      f"{fmt(sl['ankle_speed_cm_s']['mean'], 1)} / sole-min {fmt(sl['sole_min_speed_cm_s']['mean'], 1)} "
      f"(simulator contact) | not comparable | |")
  a("")
  a("Foot slip is not like-for-like: the reference has no simulator, so its number comes from "
    "`g1/eval_g1_motion.py` on `_g1_fixed.pkl` (30 fps, before the 12 Hz filter) with EDGE's predicted "
    "contacts. The executed number uses MuJoCo's foot/floor contacts.\n")
  a("### Wrong-song calibration of the beat alignment score (kinematic->music)\n")
  a("| music | reference | executed (clean) |")
  a("|---|---|---|")
  a(f"| **{R['song']} (true alignment, {R['audio']['offset_s']} s)** | {fmt(cr['bas']['kin_to_music'], 4)} | "
    f"{fmt(ce['bas']['kin_to_music'], 4)} |")
  for spec in cr["bas_calibration"]:
    a(f"| {spec} | {fmt(cr['bas_calibration'][spec]['kin_to_music'], 4)} | "
      f"{fmt(ce['bas_calibration'][spec]['kin_to_music'], 4)} |")
  a("")
  a("## Completion\n")
  a("| protocol | episodes | success rate | completion fraction (mean / min) | first-termination conditions |")
  a("|---|---|---|---|---|")

  def crow(name, S):
    if not S:
      return
    a(f"| {name} | {S['n_episodes']} | {fmt(S['success_rate'], 2)} ({S['n_success']}/{S['n_episodes']}) | "
      f"{fmt(S['completion_fraction']['mean'], 3)} / {fmt(S['completion_fraction']['min'] if 'min' in S['completion_fraction'] else None, 3)} | "
      f"{', '.join(f'{k}: {v}' for k, v in S['first_term_conditions'].items()) or 'none'} |")

  def single(S):
    return {"n_episodes": 1, "success_rate": float(S["completed"]), "n_success": int(S["completed"]),
            "completion_fraction": {"mean": S["completion_fraction"]},
            "first_term_conditions": {k: 1 for k in S["first_term_conditions"]}}

  crow("clean (no randomization)", single(c))
  crow("randomized (training reset perturbation + startup DR + obs noise)", rs)
  if R.get("openloop"):
    crow("open-loop reference joint targets (no policy, clean)", single(R["openloop"]["summary"]))
  if R.get("push"):
    crow("randomized + training pushes", R["push"]["set"])
  for f, S in (R.get("friction") or {}).items():
    crow(f"clean, foot friction = {f}", single(S))
  a("")
  if R.get("openloop"):
    o = R["openloop"]["summary"]
    a(f"Open-loop baseline: the robot survives **{fmt(o['first_term_time_s'] or (o['frames_scored'] / R['fps']), 2)} s** "
      f"of {fmt((R['n_frames'] - 1) / R['fps'], 2)} s"
      f"{' (terminated by ' + ', '.join(o['first_term_conditions']) + ')' if o['first_term_conditions'] else ' (completed)'}.\n")
  if R.get("push"):
    for e in R["push"]["episodes"]:
      if not e["completed"]:
        a(f"Push run seed {e['seed']} terminated at {fmt(e['first_term_time_s'], 2)} s by "
          f"{', '.join(e['first_term_conditions'])}"
          + (". That is before the first push could fire: mjlab resamples the push timer at every reset, to "
             "1-3 s, so the cause is that seed's reset perturbation and DR draw, not a push."
             if e["first_term_time_s"] is not None and e["first_term_time_s"] < 1.0 else ".") + "\n")
  times = [x for x in rs["first_term_times_s"] if x is not None]
  if times:
    a(f"Randomized first-termination times [s]: {', '.join(f'{x:.2f}' for x in sorted(times))}.\n")
  a("Fraction of randomized episodes that complete the clip with the body-mean local position error "
    "never above the threshold: " + ", ".join(f"{X:g} cm: {fmt(rs[f'frac_under_{X:g}cm'], 2)}"
                                              for X in cfg["pos_error_thresholds_cm"]) + ".\n")
  a("## Tracking error\n")
  a("Statistics over every scored frame (frames 1 to the end, or to the first termination). "
    "*Averaged* = the mean over the 14 tracked bodies (29 joints for joint error) at each frame, then the "
    "statistic over frames. Randomized = all frames of all "
    f"{R['randomized']['set']['n_episodes']} episodes pooled.\n")
  a("| error | run | mean | median | p95 | max |")
  a("|---|---|---|---|---|---|")
  for key, lab in (("pos_local_cm", "position, local [cm]"), ("pos_global_cm", "position, global [cm]"),
                   ("ori_local_deg", "orientation, local [deg]"), ("ori_global_deg", "orientation, global [deg]"),
                   ("linvel_global", "linear velocity, global [m/s]"), ("linvel_local", "linear velocity, local [m/s]"),
                   ("acc_global", "acceleration, global [m/s^2] (2nd priority)")):
    for run_name, block in (("clean", R["clean"]["tracking"]), ("randomized", R["randomized"]["tracking"])):
      st = block[key]["body_mean"]
      a(f"| {lab} | {run_name} | {fmt(st['mean'])} | {fmt(st['median'])} | {fmt(st['p95'])} | {fmt(st['max'])} |")
  for run_name, block in (("clean", R["clean"]["tracking"]), ("randomized", R["randomized"]["tracking"])):
    st = block["joint_abs_deg"]["joint_mean"]
    a(f"| joint position, mean over joints [deg] | {run_name} | {fmt(st['mean'])} | {fmt(st['median'])} | {fmt(st['p95'])} | {fmt(st['max'])} |")
  if c.get("dtw_pos_global_cm") is not None:
    a(f"\nTime-warped (DTW) global position error, clean run: {fmt(c['dtw_pos_global_cm'])} cm "
      f"(band ±{c['dtw_band_frames']} frames) vs {fmt(R['clean']['tracking']['pos_global_cm']['body_mean']['mean'])} cm without warping.\n")
  a("\n### Per body (local position error, cm)\n")
  a("| body | clean mean | clean p95 | clean max | randomized mean | randomized p95 | randomized max |")
  a("|---|---|---|---|---|---|---|")
  for b in R["tracked_bodies"]:
    s1 = R["clean"]["tracking"]["pos_local_cm"]["per_body"][b]
    s2 = R["randomized"]["tracking"]["pos_local_cm"]["per_body"][b]
    a(f"| {b} | {fmt(s1['mean'])} | {fmt(s1['p95'])} | {fmt(s1['max'])} | {fmt(s2['mean'])} | {fmt(s2['p95'])} | {fmt(s2['max'])} |")
  a("\nFull per-body tables (global position, orientation, velocity, per-joint error) are in `metrics.json`.\n")
  a("## Per-body error against training iteration\n")
  a("Training logged no per-body error (WandB and TensorBoard hold only body averages), so the clean "
    "evaluation was rerun on every saved checkpoint. See `plots/per_body_error_vs_iteration.png`.\n")
  a("| iteration | completed | completion | body-mean local pos err [cm] | legs | arms | torso |")
  a("|---|---|---|---|---|---|---|")
  for s in R.get("sweep", []):
    g = s["group_pos_local_cm"]
    a(f"| {s['iteration']} | {fmt(s['completed'])} | {fmt(s['completion_fraction'], 3)} | {fmt(s['pos_local_cm_mean'])} | "
      f"{fmt(g['legs'])} | {fmt(g['arms'])} | {fmt(g['torso'])} |")
  a("")
  a("## Effort and smoothness (executed)\n")
  a("| metric | clean | randomized (mean over episodes) |")
  a("|---|---|---|")
  a(f"| action rate (sum over joints of squared action change per step), mean | {fmt(c['action_rate_l2']['mean'], 3)} | "
    f"{fmt(rmean(lambda s: s['action_rate_l2']['mean']), 3)} |")
  a(f"| mean abs torque [Nm] | {fmt(c['torque_mean_abs'])} | {fmt(rs['torque_mean_abs']['mean'])} |")
  a(f"| peak abs torque [Nm] | {fmt(c['torque_peak_abs'])} | {fmt(rs['torque_peak_abs']['mean'])} |")
  a(f"| highest peak torque / limit over joints | {fmt(c['torque_peak_ratio'], 3)} | |")
  a(f"| substeps with any joint at its torque limit | {fmt(c['torque_sat_frac_any_joint'], 4)} | "
    f"{fmt(rs['torque_sat_frac_any_joint']['mean'], 4)} |")
  a(f"| joint-substeps at the torque limit (mean over joints) | {fmt(c['torque_sat_frac_mean_joint'], 4)} | |")
  top = sorted(c["torque_sat_frac_per_joint"].items(), key=lambda kv: -kv[1])[:5]
  a(f"\nJoints most often at the torque limit (clean): " + ", ".join(f"{j} {v:.4f}" for j, v in top) + ".\n")
  a("## Training cost\n")
  T = R["training"]
  a(f"- Wall clock: {fmt(T.get('wall_clock_h'), 2)} h ({T.get('started')} to {T.get('finished')}), "
    f"{T.get('iterations_logged')} iterations, {T['num_envs']} environments, {T['num_steps_per_env']} steps per env per iteration.")
  a(f"- Environment steps: {T.get('env_steps_total'):,} total; {fmt(T['env_steps_per_s']['mean'], 0)} steps/s "
    f"(mean of the logged Perf/total_fps), {fmt(T.get('env_steps_per_s_overall'), 0)} steps/s overall including logging and checkpointing.")
  fc = R.get("first_checkpoint_completing_clean")
  a(f"- First checkpoint that completes the clean run: **{fc if fc is not None else 'none'}**"
    + (f" (≈ {fc / max(T.get('iterations_logged', 1), 1) * T.get('wall_clock_h', 0):.2f} h into training)." if fc is not None else "."))
  comp = [r for r in R["training_runs"] if r["comparable"]]
  a(f"- Training seeds: {len(comp)} finished run(s) with this budget "
    f"(seed {', '.join(str(r['seed']) for r in comp)}). "
    + ("Only one training seed exists, so there is no seed spread to report. " if len(comp) == 1 else "")
    + "Other runs in the folder, not counted: " + (", ".join(
      f"`{r['dir']}` ({r['max_iterations']} iterations planned, {r['num_envs']} envs, last checkpoint {r['last_iteration']})"
      for r in R["training_runs"] if not r["comparable"]) or "none") + ".\n")
  a("## Metric self-tests\n")
  a("| test | result | detail |")
  a("|---|---|---|")
  for t in R["self_tests"]:
    a(f"| {t['test']} | {'PASS' if t['passed'] else '**FAIL**'} | {t['detail']} |")
  a("\n## Run-to-run variation of the clean run\n")
  a(f"MuJoCo Warp on the GPU is not bit-deterministic (mujoco_warp issue 562, noted in mjlab's `seed_rng`). "
    f"Two identical clean rollouts (the second with video rendering) agree exactly at first and then drift "
    f"apart: max difference {R['determinism']['max_body_pos_diff_m'] * 1000:.3g} mm in body position and "
    f"{np.degrees(R['determinism']['max_joint_pos_diff_rad']):.3g} deg in joint position over the clip. The "
    f"clean run was therefore repeated {R['clean_repeats']['n']} times. The tables above show repeat 0. The "
    "spread over all repeats is:\n")
  a("| metric | min | mean | max |")
  a("|---|---|---|---|")
  for k, st in R["clean_repeats"]["spread"].items():
    if k == "completed":
      continue
    a(f"| {k} | {fmt(st['min'], 4)} | {fmt(st['mean'], 4)} | {fmt(st['max'], 4)} |")
  a(f"\nCompleted the clip in {sum(R['clean_repeats']['spread']['completed'])}/{R['clean_repeats']['n']} repeats. "
    "The same applies to the randomized episodes: a seed fixes the perturbation, DR draw and observation noise, "
    "but rerunning that seed does not reproduce the trajectory bit for bit.\n")
  a(DEFINITIONS.format(**R["defs"]))
  a("## Thresholds and where they come from\n")
  a("| parameter | value | source |")
  a("|---|---|---|")
  for name, val, src in R["thresholds"]:
    a(f"| {name} | {val} | {src} |")
  a("")
  a("## Provenance\n")
  P = R["provenance"]
  a(f"- **Motion version**: {R['motion_version']}")
  a(f"- Reference file `{R['motion_file']}`, sha256 `{P['sha256']['motion_npz'][:16]}…`; source `{P['source_pkl']}` sha256 `{P['sha256']['g1_fixed_pkl'][:16]}…`.")
  a(f"- Checkpoint `{R['checkpoint']}`, sha256 `{P['sha256']['checkpoint'][:16]}…`.")
  a(f"- Audio `{R['audio']['path']}` (sha256 `{P['sha256']['audio'][:16]}…`): motion frame 0 = {R['audio']['offset_s']} s into the song. {R['audio']['how']}")
  a(f"- Randomized seeds: {R['randomized']['set']['seeds'][0]}..{R['randomized']['set']['seeds'][-1]} (seed i -> torch, numpy, python and warp RNGs reseeded before the episode's startup DR and reset)."
    + (f" Push seeds: {R['push']['set']['seeds'][0]}..{R['push']['set']['seeds'][-1]}." if R.get("push") else ""))
  a(f"- Training config check: {R['config_check']}")
  a(f"- Edge repo commit `{P['edge_commit']}`{' (uncommitted changes: ' + ', '.join(P['edge_dirty']) + ')' if P['edge_dirty'] else ''}; "
    f"mjlab commit `{P['mjlab_commit']}`{' (dirty)' if P['mjlab_dirty'] else ' (clean)'}; training-time mjlab commit `{P.get('mjlab_commit_at_training')}`.")
  a("- Library versions: " + ", ".join(f"{k} {v}" for k, v in P["versions"].items()) + f"; GPU {P['gpu']}.")
  a("")
  a("## Not computed, or computed differently than specified\n")
  for x in R["caveats"]:
    a(f"- {x}")
  a("")
  (out / "report.md").write_text("\n".join(L))


DEFINITIONS = """## Metric definitions

All metrics run on the 50 Hz policy grid. Frame *k* of an episode is the robot state after env step *k*,
compared with reference frame *k* (checked at run time: mjlab scores step *k* against command frame *k*).
Frame 0 is the reset state, which `MotionCommand` writes from reference frame 0, with joint positions
clipped to the soft joint limits (90 % of the range). It is excluded from all error statistics.
Unless stated otherwise, tracking errors cover the 14 bodies the task tracks: {tracked}.

- **Completion.** An episode succeeds when it reaches the last clip frame ({last_frame}, {clip_s:.2f} s)
  without any training termination term firing (`anchor_pos`, `anchor_ori`, `ee_body_pos`; `time_out` is
  disabled by setting the episode length to 1e9 s). The *first-termination time* is k/50 s for the step k
  where a term first fires. The *completion fraction* is (last scored frame)/(last clip frame). The
  terminal frame is kept and scored. Nothing after it is recorded, because `auto_reset` is off.
- **Local error (training reward frame).** This is mjlab `MotionCommand.update_relative_body_poses`. The
  reference is translated so its anchor (`torso_link`) sits at the robot anchor's x, y, with the
  reference anchor's own z, then turned by the yaw of q_robot_anchor · q_ref_anchor⁻¹. Position error is
  ‖p_robot − p_ref,rel‖ (cm). Orientation error is the rotation angle between q_robot and q_ref,rel (deg).
  Self-test 5 checks this numpy implementation against the values mjlab computed inside the env.
- **Global error.** The same quantities in the world frame, with no re-anchoring (root drift included).
- **Linear velocity error.** Global: ‖v_robot − v_ref‖ (m/s), the training term `motion_body_lin_vel`.
  Local: the reference velocity is first turned by the same heading difference.
- **Joint position error.** |q_robot − q_ref| per joint (deg). The L2 norm over joints in rad, the
  training log's `error_joint_pos`, is in `metrics.json`.
- **Acceleration error (second priority).** ‖a_robot − a_ref‖, with a = central difference of body linear
  velocity (m/s²).
- **DTW error (second priority).** Dynamic time warping between the reference and executed sequences of
  the 14 global body positions. The frame cost is the mean body distance (cm), the Sakoe-Chiba band is ±25
  frames (0.5 s), and the steps are (1,0), (0,1), (1,1). The result is the mean cost along the optimal path.
- **Stays under X cm.** The episode completes and the per-frame body-mean local position error is ≤ X on
  every frame.
- **PFC.** EDGE `eval/eval_pfc.py`, unchanged except for the body mapping. SMPL pelvis → G1 `pelvis`;
  SMPL ankles (7, 8) → `left/right_ankle_roll_link` origin; SMPL feet/toes (10, 11) → the front tip of the
  G1 sole, i.e. the front-most end point of the `*_foot*_collision` capsules, at {toe} m in the
  `ankle_roll_link` frame. DT = 1/50. Up axis z. Root acceleration normalised by its maximum over the clip.
  Foot speeds are per frame, as in EDGE, so PFC depends on the frame rate: the same motion scores
  (30/50)² = 0.36 × its 30 fps value. These numbers are therefore not comparable with EDGE's paper
  values, which are at 30 fps. For context, the source EDGE motion (SMPL joints, 30 fps, EDGE's own code
  unchanged) scores {edge_pfc}.
- **Foot slip (executed).** For each foot, the frames where MuJoCo reports contact between that foot's
  subtree and the terrain (a passive contact sensor added for the evaluation). Two speeds are reported:
  the horizontal speed of the `ankle_roll_link` origin (the literal spec), and the smaller of the
  horizontal speeds of the sole's heel and toe tips (rigid-body velocity v + ω × r). The second does not
  count a foot pivoting on its heel or toe as slip. Also reported: the fraction of contact frames above
  {slip} cm/s.
- **Foot slip (reference).** `g1/eval_g1_motion.py` (the existing checks script) run unchanged on
  `<song>_g1_fixed.pkl`: G1 heel/toe speed while EDGE's predicted contact is > 0.95.
- **Beat alignment.**
  - Music beats: librosa {librosa} `beat_track` on the onset envelope of the song from {offset} s for the
    clip's duration, at sr = 50 × 512 so that one librosa frame is one motion frame (MINT
    `tools/preprocessing.py`, with tightness 100 and start_bpm = librosa's own tempo estimate).
  - Kinematic beats: local minima (`argrelextrema`, `np.less`) of the kinetic velocity, which is the mean
    over all 30 G1 bodies of the per-frame displacement norm, Gaussian-smoothed with σ = {ksig:.3f} s
    ({ksig_f:.2f} frames). This is Bailando `utils/beat_align_score.py`'s σ = 5 frames at 60 fps.
  - Score (primary, as specified and as in the AIST++ paper): the mean over kinematic beats of
    exp(−d²/2σ²), where d is the distance in frames to the nearest music beat and σ = 0.05 s = {bsig:.1f}
    frames.
  - Also reported: Bailando's `BA()` direction, the mean over music beats of the same kernel to the
    nearest kinematic beat. EDGE's paper numbers use this direction.
  - The file `tools/calculate_beat_scores.py` doesn't exist in google-research/mint (checked against the
    repo tree and its history), so the port follows the paper definition, Bailando's implementation and
    MINT's audio preprocessing.
- **Lag.** The integer shift in [−25, 25] frames (±0.5 s) that maximises the Pearson correlation between
  the reference and executed unsmoothed kinetic velocity curves over their overlap. Positive means the
  executed motion is late. The sub-frame value comes from a parabolic fit through the peak.
- **Joint jerk.** |Δ³q|·fps³ per joint and frame (rad/s³), mean and 95th percentile over joints and frames.
- **Action rate.** Σ_j (a_k − a_{{k−1}})² per step over the raw policy actions, for steps 2 onward. Step 1
  is excluded because mjlab's last action is zero at reset. This matches the training term `action_rate_l2`.
- **Torque.** MuJoCo `actuator_force`, recorded at every physics substep (4 per policy step, 200 Hz).
  *At the limit* means |τ| ≥ {tq} × the actuator `forcerange`.
- **Near a joint limit.** Within {jm:g}° of the MuJoCo joint range (hard limit).
  **Near a velocity limit** means |q̇| ≥ {vm:g} × the motor velocity limit from mjlab's `g1_constants.py`.
  mjlab doesn't enforce these limits in simulation.

"""


# =============================================================================== main


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run", type=Path)
  ap.add_argument("--train_dir", type=Path, help="training log folder (default: the most complete one)")
  ap.add_argument("--checkpoint", type=Path, help="default: the last checkpoint of the training run")
  ap.add_argument("--out", type=Path)
  ap.add_argument("--no-second", action="store_true", help="skip pushes, friction and DTW")
  ap.add_argument("--rescore", action="store_true", help="reuse saved rollouts in raw/")
  ap.add_argument("--episodes", type=int, help="override the number of randomized episodes")
  ap.add_argument("--frames", type=int, help="debug: stop every rollout after this many steps")
  a = ap.parse_args()
  cfg = EvalConfig()
  if a.episodes:
    cfg.n_random_episodes = a.episodes
  run = a.run.resolve()
  edge_pkl = [p for p in run.glob("*.pkl") if "_g1" not in p.name][0]
  song = edge_pkl.stem
  train_runs = all_training_runs(run)
  train_dir = (a.train_dir.resolve() if a.train_dir else
               run / "rl" / "g1_tracking" / max(train_runs, key=lambda r: len(r["checkpoints"]))["dir"])
  this = next(r for r in train_runs if r["dir"] == train_dir.name)
  cfg.task = this["task"]
  for r in train_runs:  # a training seed = a finished run of the same task and budget with its own seed
    r["comparable"] = (r["complete"] and r["max_iterations"] == this["max_iterations"]
                       and r["num_envs"] == this["num_envs"] and r["task"] == this["task"]
                       and (r["seed"] != this["seed"] or r["dir"] == this["dir"]))
  complete = [r for r in train_runs if r["comparable"]]
  ckpts = sorted(train_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  ckpt = (a.checkpoint.resolve() if a.checkpoint else ckpts[-1])
  out = (a.out or run / "eval" / train_dir.name).resolve()
  raw, logs = out / "raw", out / "logs"
  raw.mkdir(parents=True, exist_ok=True)
  logs.mkdir(exist_ok=True)
  print(f"evaluating {train_dir.name} / {ckpt.name} -> {out}")
  if len(complete) > 1:
    print(f"NOTE: {len(complete)} complete training runs found; this evaluates {train_dir.name}. "
          "Run once per training dir (--train_dir) for the seed spread.")

  R = {"song": song, "checkpoint": str(ckpt), "config": cfg.to_dict(),
       "generated": dt.datetime.now().isoformat(timespec="seconds"), "training_runs": train_runs}
  caveats = []
  RO = [sys.executable, HERE / "rollout.py"]
  common = ["--run", run, "--train_dir", train_dir] + (["--frames", a.frames] if a.frames else [])

  # ---- 0. config check
  txt = run_cmd(RO + ["verify"] + common, logs / "verify.log")
  match = "CONFIG_MATCH" in txt
  R["config_check"] = ("the env config rebuilt from mjlab's registry is identical to the run's "
                       "params/env.yaml (YAML dump compared as text)" if match else
                       "MISMATCH, see logs/verify.log")
  if not match:
    raise SystemExit("rebuilt training config differs from params/env.yaml; see logs/verify.log")

  # ---- 1. rollouts
  def ro(mode, name, *extra):
    p = raw / f"{name}.npz"
    if a.rescore and p.exists():
      return p
    run_cmd(RO + [mode] + common + ["--out", p] + [str(x) for x in extra], logs / f"{name}.log")
    return p

  seeds = f"{cfg.random_seed_base}..{cfg.random_seed_base + cfg.n_random_episodes - 1}"
  vid_raw = out / "clean_run_noaudio.mp4"
  p_clean = ro("clean", "clean", "--checkpoint", ckpt)
  p_clean_v = ro("clean", "clean_video", "--checkpoint", ckpt, "--video", vid_raw)
  p_rep = ro("clean", "clean_repeats", "--checkpoint", ckpt, "--seeds", f"1..{cfg.clean_repeats - 1}") \
    if cfg.clean_repeats > 1 else None
  p_rand = ro("random", "randomized", "--checkpoint", ckpt, "--seeds", seeds)
  p_open = ro("openloop", "openloop")
  p_sweep = ro("sweep", "sweep") if cfg.sweep_all_checkpoints else None
  p_push = p_fric = None
  if not a.no_second:
    n_push = min(cfg.push_episodes, cfg.n_random_episodes) if a.episodes else cfg.push_episodes
    ps = f"{cfg.random_seed_base + 10000}..{cfg.random_seed_base + 10000 + n_push - 1}"
    p_push = ro("push", "push", "--checkpoint", ckpt, "--seeds", ps)
    p_fric = ro("friction", "friction", "--checkpoint", ckpt, "--friction",
                ",".join(str(f) for f in cfg.friction_values))

  meta, clean_eps = load_npz(p_clean)
  motion_file = Path(str(meta["motion_file"]))
  ref = load_reference(motion_file)
  fps = ref["fps"]
  T = len(ref["body_pos"])
  R.update(fps=fps, n_frames=T, motion_file=str(motion_file))

  # ---- 2. audio + music beats
  wav = EDGE_ROOT / "custom_music" / f"{song}.wav"
  offset, how = audio_offset(run, song, wav)
  R["audio"] = {"path": str(wav), "offset_s": offset, "how": how}
  beats_json = raw / "music_beats.json"
  if not (a.rescore and beats_json.exists()):
    cmd = [EDGE_PY, HERE / "music_beats.py", beats_json, "--song", wav, "--offset", offset, "--fps", fps,
           "--n_frames", T, "--hop", cfg.music_hop_length, "--tightness", cfg.music_beat_tightness]
    for s in cfg.calibration_songs:
      cmd += ["--extra", s]
    run_cmd(cmd, logs / "music_beats.log")
  beats = json.loads(beats_json.read_text())
  R["music"] = {"song": {k: v for k, v in beats["song"].items()},
                "calibration": {k: {kk: vv for kk, vv in v.items() if kk != "beat_frames"}
                                for k, v in beats["calibration"].items()}}

  ctx = Ctx(ref, meta, cfg, beats)

  # ---- 3. video with audio
  final_vid = out / "clean_run.mp4"
  if vid_raw.exists():
    dur = (T - 1) / fps
    subprocess.run([str(FFMPEG), "-loglevel", "error", "-y", "-i", str(vid_raw), "-ss", str(offset),
                    "-t", f"{dur:.3f}", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", str(final_vid)], check=True)
    vid_raw.unlink()

  # ---- 4. score
  print("scoring ...", flush=True)
  clean_summ, clean_te = score_episode(ctx, clean_eps[0], want_dtw=not a.no_second)
  clean_te["_exe_body_pos"] = np.asarray(clean_eps[0]["body_pos"], dtype=np.float64)
  R["tracked_bodies"] = ctx.tracked
  R["clean"] = {"summary": clean_summ, "tracking": pooled_tracking(ctx, [clean_te])}
  _, vid_eps = load_npz(p_clean_v)
  R["determinism"] = {
    "max_body_pos_diff_m": float(np.abs(vid_eps[0]["body_pos"][:len(clean_eps[0]["body_pos"])] -
                                        clean_eps[0]["body_pos"][:len(vid_eps[0]["body_pos"])]).max()),
    "max_joint_pos_diff_rad": float(np.abs(vid_eps[0]["joint_pos"][:len(clean_eps[0]["joint_pos"])] -
                                           clean_eps[0]["joint_pos"][:len(vid_eps[0]["joint_pos"])]).max()),
    "same_length": len(vid_eps[0]["body_pos"]) == len(clean_eps[0]["body_pos"])}

  rep = [clean_summ]
  if p_rep:
    _, rep_eps = load_npz(p_rep)
    rep += [score_episode(ctx, e)[0] for e in rep_eps]
  keys = ["completion_fraction", "pos_local_cm_mean", "pos_global_cm_mean", "ori_local_deg_mean",
          "ori_global_deg_mean", "linvel_global_mean", "joint_abs_deg_mean", "lag_ms", "torque_mean_abs",
          "torque_sat_frac_any_joint"]
  spread = {k: M.stats([r[k] for r in rep if r[k] is not None]) for k in keys}
  spread["pfc_exec"] = M.stats([r["exec"]["pfc"] for r in rep])
  spread["bas_exec_kin_to_music"] = M.stats([r["exec"]["bas"]["kin_to_music"] for r in rep])
  spread["slip_left_sole_min_mean"] = M.stats([r["foot_slip"]["left"]["sole_min_speed_cm_s"]["mean"] for r in rep])
  spread["slip_right_sole_min_mean"] = M.stats([r["foot_slip"]["right"]["sole_min_speed_cm_s"]["mean"] for r in rep])
  spread["completed"] = [r["completed"] for r in rep]
  R["clean_repeats"] = {"n": len(rep), "spread": spread}

  _, rand_eps = load_npz(p_rand)
  rand = [score_episode(ctx, e) for e in rand_eps]
  rand_summs, rand_tes = [r[0] for r in rand], [r[1] for r in rand]
  R["randomized"] = {"episodes": rand_summs, "set": episode_set_summary(ctx, rand_summs),
                     "tracking": pooled_tracking(ctx, rand_tes)}

  _, open_eps = load_npz(p_open)
  o_summ, o_te = score_episode(ctx, open_eps[0])
  R["openloop"] = {"summary": o_summ, "tracking": pooled_tracking(ctx, [o_te])}

  sweep = []
  if p_sweep:
    _, sw_eps = load_npz(p_sweep)
    for e in sw_eps:
      s, te = score_episode(ctx, e)
      it = int(Path(str(e["checkpoint"])).stem.split("_")[1])
      per = {b: float(te["pos_local_cm"][1:, i].mean()) for i, b in enumerate(ctx.tracked)}
      sweep.append({"iteration": it, "completed": s["completed"], "completion_fraction": s["completion_fraction"],
                    "first_term_time_s": s["first_term_time_s"], "first_term_conditions": s["first_term_conditions"],
                    "pos_local_cm_mean": s["pos_local_cm_mean"], "pos_global_cm_mean": s["pos_global_cm_mean"],
                    "ori_local_deg_mean": s["ori_local_deg_mean"], "joint_abs_deg_mean": s["joint_abs_deg_mean"],
                    "per_body_pos_local_cm": per,
                    "per_body_pos_global_cm": {b: float(te["pos_global_cm"][1:, i].mean()) for i, b in enumerate(ctx.tracked)},
                    "per_body_ori_local_deg": {b: float(te["ori_local_deg"][1:, i].mean()) for i, b in enumerate(ctx.tracked)},
                    "group_pos_local_cm": {g: float(np.mean([per[b] for b in bs])) for g, bs in cfg.body_groups.items()}})
    sweep.sort(key=lambda s: s["iteration"])
  R["sweep"] = sweep
  firsts = [s["iteration"] for s in sweep if s["completed"]]
  R["first_checkpoint_completing_clean"] = firsts[0] if firsts else None
  if sweep:
    later_fail = [s["iteration"] for s in sweep if firsts and s["iteration"] > firsts[0] and not s["completed"]]
    if later_fail:
      caveats.append(f"Checkpoints after the first clean completion that fail the clean run again: {later_fail}.")

  push_summs = None
  if p_push:
    _, push_eps = load_npz(p_push)
    pr = [score_episode(ctx, e) for e in push_eps]
    push_summs = [x[0] for x in pr]
    R["push"] = {"episodes": push_summs, "set": episode_set_summary(ctx, push_summs),
                 "tracking": pooled_tracking(ctx, [x[1] for x in pr])}
  if p_fric:
    fm, fr_eps = load_npz(p_fric)
    R["friction"] = {}
    for f, e in zip(fm["frictions"], fr_eps):
      s, _ = score_episode(ctx, e)
      R["friction"][f"{f:g}"] = s

  # reference-only extras
  R["reference_full"] = motion_only(ctx, ref)
  R["reference_foot_slip"] = reference_slip(run, song, logs)
  R["edge_source_pfc_30fps"] = edge_source_pfc(edge_pkl)

  # ---- 5. self-tests
  print("self-tests:")
  R["self_tests"] = self_tests(ctx, clean_eps[0])

  # ---- 6. training info
  R["training"] = training_info(run, train_dir)
  if R["training"].get("per_body_scalars"):
    caveats.append("Training logs do contain per-body scalars; the per-iteration plot still uses the clean-run sweep.")

  # ---- 7. provenance
  R["motion_version"] = motion_version(run, song, logs)
  R["provenance"] = provenance(run, song, motion_file, ckpt, wav, train_dir)

  # ---- 8. caveats
  caveats += [
    "Only one complete training seed exists (seed 42), so no seed spread can be reported." if len(complete) == 1 else
    f"{len(complete)} complete training seeds exist; this report covers {train_dir.name} only.",
    "Training scored the stochastic policy on 10 s windows with adaptive start frames and pushes; this "
    "evaluation scores the deterministic policy on the full clip from frame 0, so its errors are not the same "
    "quantity as the training log's Metrics/motion/* values (those are listed in metrics.json under training.final_logged).",
    "Foot slip on the reference cannot use simulator contacts (there is no simulation). The reference number is "
    "the existing checks script on the 30 fps foot-fixed retarget with EDGE contact labels, so reference and "
    "executed slip are reported side by side but are different measurements.",
    "Action rate and torque exist only for the executed motion (the reference has no actions; inverse dynamics "
    "of the reference was not requested and was not computed).",
    "PFC is computed at 50 fps on G1 bodies; it is not comparable with EDGE's published (30 fps, SMPL) values. "
    "The source EDGE motion's own 30 fps SMPL PFC is given for context only.",
    "The specified MINT script tools/calculate_beat_scores.py does not exist in google-research/mint; the beat "
    "alignment port follows the AIST++ paper definition (kinematic->music) and reports Bailando's direction too.",
    "MINT takes start_bpm from the AIST++ file name; here it is librosa's own tempo estimate of the clip.",
    "Velocity limits are the motor limits in mjlab's g1_constants.py; for the ankle and waist-pitch/roll "
    "linkages mjlab assumes a 1:1 ratio, and the 5020's 37 rad/s is used for them.",
  ]
  R["caveats"] = caveats
  R["thresholds"] = thresholds_table(cfg)
  R["defs"] = dict(tracked=", ".join(f"`{b}`" for b in ctx.tracked), last_frame=T - 1, clip_s=(T - 1) / fps,
                   toe=np.round(ctx.toe["left"], 4).tolist(), edge_pfc=fmt(R["edge_source_pfc_30fps"], 4),
                   slip=cfg.slip_speed_threshold_cm_s, librosa=beats.get("librosa_version"), offset=offset,
                   ksig=cfg.kinetic_smooth_sigma_s, ksig_f=cfg.kinetic_smooth_sigma_s * fps,
                   bsig=cfg.beat_sigma_s * fps, tq=cfg.torque_limit_frac, jm=cfg.joint_limit_margin_deg,
                   vm=1 - cfg.velocity_limit_margin_frac)

  # ---- 9. outputs
  plot_all(out, ctx, clean_summ, clean_te, rand_summs, rand_tes, sweep, o_summ, push_summs)
  write_csv(out / "episodes.csv", R)
  dump = {k: v for k, v in R.items() if k not in ("defs",)}
  for blk in ("clean",):
    dump[blk]["summary"]["exec"].pop("kinematic_beats", None)
    dump[blk]["summary"]["ref_same_segment"].pop("kinematic_beats", None)
  for blk in ("randomized", "push"):
    if blk in dump:
      for s in dump[blk]["episodes"]:
        s["exec"].pop("kinematic_beats", None)
        s["ref_same_segment"].pop("kinematic_beats", None)
  for s in (dump.get("friction") or {}).values():
    s["exec"].pop("kinematic_beats", None)
    s["ref_same_segment"].pop("kinematic_beats", None)
  dump["openloop"]["summary"]["exec"].pop("kinematic_beats", None)
  dump["openloop"]["summary"]["ref_same_segment"].pop("kinematic_beats", None)
  dump["reference_full"].pop("kinematic_beats", None)
  (out / "metrics.json").write_text(json.dumps(clean_json(json.loads(json.dumps(dump, default=jdefault))), indent=1))
  write_report(out, R)
  print(f"\ndone -> {out}")
  n_fail = sum(not t["passed"] for t in R["self_tests"])
  print(f"self-tests: {len(R['self_tests']) - n_fail} passed, {n_fail} failed")


def write_csv(path, R):
  rows = []
  for proto in ("clean", "randomized", "openloop", "push", "friction"):
    if proto not in R or R[proto] is None:
      continue
    if proto in ("clean", "openloop"):
      eps = [("", R[proto]["summary"])]
    elif proto == "friction":
      eps = list(R["friction"].items())
    else:
      eps = [("", s) for s in R[proto]["episodes"]]
    for tag, s in eps:
      rows.append({
        "protocol": proto, "friction": tag, "seed": s["seed"], "checkpoint": Path(s["checkpoint"]).name or "open-loop",
        "completed": int(s["completed"]), "completion_fraction": s["completion_fraction"],
        "first_term_time_s": s["first_term_time_s"], "first_term_conditions": "|".join(s["first_term_conditions"]),
        "frames_scored": s["frames_scored"],
        **{f"{k}_mean": s[f"{k}_mean"] for k in ("pos_local_cm", "pos_global_cm", "ori_local_deg", "ori_global_deg",
                                                  "linvel_global", "linvel_local", "acc_global", "joint_abs_deg")},
        "pos_local_body_mean_max_cm": s["pos_local_body_mean_max_cm"],
        **{k: int(v) for k, v in s.items() if k.startswith("under_")},
        "lag_ms": s["lag_ms"], "lag_ms_subframe": s["lag_ms_subframe"],
        "pfc_exec": s["exec"]["pfc"], "pfc_ref_segment": s["ref_same_segment"]["pfc"],
        "bas_kin_to_music_exec": s["exec"]["bas"]["kin_to_music"],
        "bas_kin_to_music_ref_segment": s["ref_same_segment"]["bas"]["kin_to_music"],
        "slip_left_ankle_mean_cm_s": s["foot_slip"]["left"]["ankle_speed_cm_s"]["mean"],
        "slip_right_ankle_mean_cm_s": s["foot_slip"]["right"]["ankle_speed_cm_s"]["mean"],
        "slip_left_sole_min_mean_cm_s": s["foot_slip"]["left"]["sole_min_speed_cm_s"]["mean"],
        "slip_right_sole_min_mean_cm_s": s["foot_slip"]["right"]["sole_min_speed_cm_s"]["mean"],
        "action_rate_mean": s["action_rate_l2"]["mean"], "torque_mean_abs": s["torque_mean_abs"],
        "torque_peak_abs": s["torque_peak_abs"], "torque_sat_frac_any_joint": s["torque_sat_frac_any_joint"],
        "joint_jerk_mean_exec": s["exec"]["joint_jerk_mean"],
        "near_pos_limit_frac_any_exec": s["exec"]["near_pos_limit_frac_any_joint"],
        "near_vel_limit_frac_any_exec": s["exec"]["near_vel_limit_frac_any_joint"],
      })
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
      w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def thresholds_table(cfg: EvalConfig):
  return [
    ("anchor_pos termination", "|z_torso - z_torso,ref| > 0.25 m", "training config (params/env.yaml)"),
    ("anchor_ori termination", "|g_z,ref - g_z| > 0.8 (projected gravity, torso frame)", "training config"),
    ("ee_body_pos termination", "|z - z_ref,rel| > 0.25 m on either ankle_roll or wrist_yaw link", "training config"),
    ("position-error thresholds", ", ".join(f"{x:g} cm" for x in cfg.pos_error_thresholds_cm),
     "chosen for this evaluation (no external standard); body-mean local error, every frame"),
    ("joint position limit margin", f"{cfg.joint_limit_margin_deg:g} deg", "same as g1/eval_g1_motion.py"),
    ("velocity limit margin", f"{cfg.velocity_limit_margin_frac:g} x limit", "chosen for this evaluation"),
    ("velocity limits", "5020: 37, 7520-14: 32, 7520-22: 20, 4010: 22 rad/s", "mjlab g1_constants.py"),
    ("torque limit", f"|tau| >= {cfg.torque_limit_frac:g} x forcerange", "forcerange from the MuJoCo model (mjlab effort_limit)"),
    ("foot slip speed", f"{cfg.slip_speed_threshold_cm_s:g} cm/s", "same as g1/eval_g1_motion.py"),
    ("beat sigma", f"{cfg.beat_sigma_s:g} s", "AIST++ / Bailando: 3 frames at 60 fps"),
    ("kinetic velocity smoothing", f"{cfg.kinetic_smooth_sigma_s:.4f} s", "Bailando: gaussian_filter sigma 5 at 60 fps"),
    ("music analysis", f"hop {cfg.music_hop_length}, sr = fps x hop, tightness {cfg.music_beat_tightness:g}", "MINT tools/preprocessing.py"),
    ("lag search window", f"±{cfg.max_lag_s:g} s", "chosen for this evaluation"),
    ("randomized episodes / seeds", f"{cfg.n_random_episodes}, seeds from {cfg.random_seed_base}", "specification"),
    ("reset perturbation (randomized)", "pose ±5 cm xy, ±1 cm z, ±0.1 rad roll/pitch, ±0.2 rad yaw; velocity as push range; joints ±0.1 rad", "training config"),
    ("startup DR (randomized)", "torso CoM ±2.5/5/5 cm, encoder bias ±0.01 rad, foot friction 0.3-1.2", "training config"),
    ("observation noise (randomized)", "training actor noise (anchor pos ±0.25, ori ±0.05, lin vel ±0.5, ang vel ±0.2, q ±0.01, qd ±0.5)", "training config"),
    ("pushes (second priority)", "every 1-3 s, velocity kick ±0.5 m/s xy, ±0.2 z, ±0.52 rad/s roll/pitch, ±0.78 yaw", "training config"),
    ("friction sweep", ", ".join(f"{f:g}" for f in cfg.friction_values), "training DR range ends and the model default"),
  ]


# =============================================================================== provenance helpers


def audio_offset(run, song, wav):
  """Find where motion frame 0 sits in the song by matching the audio of EDGE's render to the wav.
  Falls back to reproducing test.py's random window from seed.txt."""
  mp4 = run / f"test_{song}.mp4"
  if mp4.exists() and FFMPEG.exists():
    with tempfile.TemporaryDirectory() as td:
      a = Path(td) / "a.wav"
      subprocess.run([str(FFMPEG), "-loglevel", "error", "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar",
                      "22050", str(a)], check=True)
      code = ("import soundfile as sf, numpy as np, scipy.signal as ss, librosa, sys\n"
              "a,sr=sf.read(sys.argv[1]); w,sw=sf.read(sys.argv[2])\n"
              "w=w if w.ndim==1 else w.mean(1); a=a if a.ndim==1 else a.mean(1)\n"
              "w=librosa.resample(w,orig_sr=sw,target_sr=sr); n=min(len(a),sr*10); a=a[:n]\n"
              "x=ss.correlate(w,a,mode='valid',method='fft')\n"
              "d=np.sqrt(ss.correlate(w**2,np.ones(n),mode='valid',method='fft'))*np.linalg.norm(a)+1e-12\n"
              "c=x/d; i=int(np.argmax(c)); print(i/sr, c[i])\n")
      r = subprocess.run([str(EDGE_PY), "-c", code, str(a), str(wav)], capture_output=True, text=True, check=True)
      off, corr = map(float, r.stdout.split()[-2:])
      off = round(off / 2.5) * 2.5 if abs(off / 2.5 - round(off / 2.5)) < 0.01 else off
      if corr > 0.9:
        return off, (f"Measured by cross-correlating the audio track of EDGE's render (test_{song}.mp4) "
                     f"with the wav: best match at {off} s, normalised correlation {corr:.3f}. EDGE's "
                     "test.py picks a random window of 2.5 s slices; with seed.txt = "
                     f"{(run / 'seed.txt').read_text().strip() or 'random'} it starts at slice {off / 2.5:g}.")
  raise SystemExit("could not determine the audio offset (no test_<song>.mp4 to match); refusing to guess")


def reference_slip(run, song, logs):
  """Existing checks script (g1/eval_g1_motion.py) on the foot-fixed retarget, in a temp folder so the
  run folder is not written to."""
  src = run / f"{song}_g1_fixed.pkl"
  if not src.exists():
    return None
  with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for f in (run / f"{song}.pkl", src):
      os.symlink(f, td / f.name)
    run_cmd(["python3", EDGE_ROOT / "g1" / "eval_g1_motion.py", td, "--g1", "g1_fixed"], logs / "eval_g1_motion.log")
    d = json.loads((td / f"{song}_g1_fixed_metrics.json").read_text())
  return d.get("contacts")


def edge_source_pfc(edge_pkl):
  """EDGE's own PFC of the source motion: SMPL full_pose at 30 fps, EDGE's joint indices."""
  import pickle
  with open(edge_pkl, "rb") as f:
    d = pickle.load(f)
  j = np.asarray(d["full_pose"], dtype=np.float64)
  return M.pfc(j[:, 0], j[:, 7], j[:, 10], j[:, 8], j[:, 11], 30.0)


def motion_version(run, song, logs):
  log = run / "train_full.log"
  smooth = None
  if log.exists():
    with open(log, "rb") as f:
      head = f.read(20000).decode("utf-8", "replace")
    for line in head.splitlines():
      if "smoothing:" in line or "building mjlab reference" in line or "Motion interpolated" in line or "[in]" in line:
        smooth = (smooth or "") + line.strip() + " / "
  return (f"foot-fixed G1 retarget `{song}_g1_fixed.pkl` (g1/fix_g1_feet.py) → g1/edge_to_mjlab.py: 12 Hz "
          f"low-pass, resampled 30 → 50 fps, frames clipped to joint limits. From the training log: {smooth or 'n/a'}")


def provenance(run, song, motion_file, ckpt, wav, train_dir):
  vers = {}
  for p in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "numpy", "scipy", "matplotlib", "tensorboard"):
    try:
      vers[p] = im.version(p)
    except Exception:
      pass
  vers["python"] = platform.python_version()
  try:
    import torch
    gpu = torch.cuda.get_device_name(0)
  except Exception:
    gpu = "n/a"
  st = subprocess.run(["git", "-C", str(EDGE_ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
  edge_dirty = [l[3:] for l in st.splitlines() if l.strip()]
  mj_dirty = bool((git(MJLAB_DIR, "status", "--porcelain") or "").strip())
  at_train = None  # mjlab's train.py stores the git state next to the W&B run's files
  for f in sorted((MJLAB_DIR / "wandb").glob("run-*/files/mjlab.diff")):
    if (f.parent / f"{train_dir.name}.onnx").exists():
      txt = f.read_text()
      at_train = txt.split("--- git commit ---")[1].split()[0]
      if "working tree clean" in txt:
        at_train += " (clean)"
  return {"edge_commit": git(EDGE_ROOT, "rev-parse", "HEAD"), "edge_dirty": edge_dirty,
          "mjlab_commit": git(MJLAB_DIR, "rev-parse", "HEAD"), "mjlab_dirty": mj_dirty,
          "mjlab_commit_at_training": at_train,
          "versions": vers, "gpu": gpu, "source_pkl": str(run / f"{song}_g1_fixed.pkl"),
          "sha256": {"motion_npz": sha256(motion_file), "g1_fixed_pkl": sha256(run / f"{song}_g1_fixed.pkl"),
                     "checkpoint": sha256(ckpt), "audio": sha256(wav)}}


if __name__ == "__main__":
  main()
