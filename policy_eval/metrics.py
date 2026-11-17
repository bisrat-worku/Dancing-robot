"""Motion metrics, pure numpy/scipy. The same functions score the reference, the executed motion and
the self-tests, so a number in the report always means one thing.

A "motion" is a dict of arrays on one frame grid (fps):
  body_pos (T,B,3) m, body_quat (T,B,4) wxyz, body_lin_vel (T,B,3) m/s, body_ang_vel (T,B,3),
  joint_pos (T,J) rad, joint_vel (T,J) rad/s, plus "body_names", "joint_names", "fps".
Quaternions are wxyz (mjlab / MuJoCo convention).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

# ------------------------------------------------------------------ quaternion helpers (wxyz)


def quat_mul(a, b):
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                   aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_apply(q, v):
  qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
  return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def yaw_quat(q):
  """Heading-only part of q (mjlab.utils.lab_api.math.yaw_quat)."""
  w, x, y, z = np.moveaxis(q, -1, 0)
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  out = np.zeros(q.shape)
  out[..., 0] = np.cos(yaw / 2)
  out[..., 3] = np.sin(yaw / 2)
  return out


def quat_angle(a, b):
  """Rotation angle (rad) between a and b, in [0, pi] (= mjlab quat_error_magnitude)."""
  d = quat_mul(a, quat_conj(b))
  return 2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


# ------------------------------------------------------------------ statistics


def stats(x):
  x = np.asarray(x, dtype=float).ravel()
  x = x[np.isfinite(x)]
  if x.size == 0:
    return {"mean": None, "median": None, "p95": None, "max": None, "min": None, "n": 0}
  return {"mean": float(x.mean()), "median": float(np.median(x)),
          "p95": float(np.percentile(x, 95)), "max": float(x.max()), "min": float(x.min()),
          "n": int(x.size)}


# ------------------------------------------------------------------ tracking error


def anchor_relative_reference(ref, exe, anchor):
  """mjlab MotionCommand.update_relative_body_poses, vectorised over frames.

  The reference is moved so its anchor sits at the robot anchor's xy (keeping the reference
  anchor's height) and turned by the heading difference between the two anchors. This is the
  frame the training rewards motion_body_pos / motion_body_ori score in.
  Returns (pos (T,B,3), quat (T,B,4), delta_quat (T,4)).
  """
  ra_p, ra_q = ref["body_pos"][:, anchor], ref["body_quat"][:, anchor]
  ea_p, ea_q = exe["body_pos"][:, anchor], exe["body_quat"][:, anchor]
  d_pos = ea_p.copy()
  d_pos[:, 2] = ra_p[:, 2]
  d_q = yaw_quat(quat_mul(ea_q, quat_conj(ra_q)))
  B = ref["body_pos"].shape[1]
  dq_b = np.repeat(d_q[:, None], B, 1)
  pos = d_pos[:, None] + quat_apply(dq_b, ref["body_pos"] - ra_p[:, None])
  quat = quat_mul(dq_b, ref["body_quat"])
  return pos, quat, d_q


def tracking_errors(ref, exe, body_idx, anchor):
  """Per-frame, per-body errors between a reference and an executed motion on the same frames.

  ref, exe: motions of equal length T (frame t of exe is scored against frame t of ref).
  body_idx: indices of the scored bodies; anchor: index of the anchor body (torso_link).
  Returns a dict of arrays, each (T, len(body_idx)) or (T, J):
    pos_local_cm   |p_exe - p_ref_rel|               (training reward frame, see above)
    pos_global_cm  |p_exe - p_ref|                   (world frame)
    ori_local_deg  angle(q_exe, q_ref_rel)
    ori_global_deg angle(q_exe, q_ref)
    linvel_global  |v_exe - v_ref|  m/s, world frame (= training reward motion_body_lin_vel)
    linvel_local   |v_exe - R(dq) v_ref|  m/s, reference velocity turned by the heading difference
    acc_global     |a_exe - a_ref|  m/s^2, a = central difference of body_lin_vel (second priority)
    joint_abs_deg  |q_exe - q_ref| per joint, degrees
    joint_l2_rad   ||q_exe - q_ref||_2 over all joints (= training log Metrics/motion/error_joint_pos)
  """
  pos_rel, quat_rel, dq = anchor_relative_reference(ref, exe, anchor)
  b = np.asarray(body_idx)
  pe, pr = exe["body_pos"][:, b], ref["body_pos"][:, b]
  qe, qr = exe["body_quat"][:, b], ref["body_quat"][:, b]
  ve, vr = exe["body_lin_vel"][:, b], ref["body_lin_vel"][:, b]
  vr_loc = quat_apply(np.repeat(dq[:, None], len(b), 1), vr)
  fps = float(ref["fps"])
  ae = np.gradient(ve, 1 / fps, axis=0) if len(ve) > 2 else np.full_like(ve, np.nan)
  ar = np.gradient(vr, 1 / fps, axis=0) if len(vr) > 2 else np.full_like(vr, np.nan)
  return {
    "pos_local_cm": 100 * np.linalg.norm(pe - pos_rel[:, b], axis=-1),
    "pos_global_cm": 100 * np.linalg.norm(pe - pr, axis=-1),
    "ori_local_deg": np.degrees(quat_angle(qe, quat_rel[:, b])),
    "ori_global_deg": np.degrees(quat_angle(qe, qr)),
    "linvel_global": np.linalg.norm(ve - vr, axis=-1),
    "linvel_local": np.linalg.norm(ve - vr_loc, axis=-1),
    "acc_global": np.linalg.norm(ae - ar, axis=-1),
    "joint_abs_deg": np.degrees(np.abs(exe["joint_pos"] - ref["joint_pos"])),
    "joint_l2_rad": np.linalg.norm(exe["joint_pos"] - ref["joint_pos"], axis=-1),
  }


def dtw_position_error(ref_pos, exe_pos, band):
  """Time-warped error (second priority): DTW between the two sequences of stacked body positions
  (frame cost = mean body distance in cm), Sakoe-Chiba band of `band` frames, steps (1,0),(0,1),
  (1,1). Returns the mean cost along the optimal path and the path length."""
  T1, T2 = len(ref_pos), len(exe_pos)
  inf = np.inf
  D = np.full((T1 + 1, T2 + 1), inf)
  D[0, 0] = 0.0
  L = np.zeros((T1 + 1, T2 + 1), dtype=np.int32)
  for i in range(1, T1 + 1):
    lo, hi = max(1, i - band), min(T2, i + band)
    c = 100 * np.linalg.norm(exe_pos[lo - 1:hi] - ref_pos[i - 1][None], axis=-1).mean(-1)
    for j in range(lo, hi + 1):
      cands = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
      k = int(np.argmin(cands))
      prev = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
      D[i, j] = cands[k] + c[j - lo]
      L[i, j] = L[prev] + 1
  if not np.isfinite(D[T1, T2]):
    return None, 0
  return float(D[T1, T2] / L[T1, T2]), int(L[T1, T2])


# ------------------------------------------------------------------ PFC (EDGE eval/eval_pfc.py)


def pfc(root_pos, l_ankle, l_toe, r_ankle, r_toe, fps, up=2):
  """EDGE's Physical Foot Contact score, line for line (Stanford-TML/EDGE eval/eval_pfc.py):
    root_v = diff(root)/DT; root_a = diff(root_v)/DT; root_a[up] = max(root_a[up], 0)
    root_a = |root_a| / max|root_a|            (normalised per clip)
    foot_v = |horizontal displacement per frame| of (l_ankle, l_toe, r_ankle, r_toe), frames 2..
    PFC = 10000 * mean_t( min(lv_ankle, lv_toe) * min(rv_ankle, rv_toe) * root_a )
  Foot speeds are per frame (not divided by DT) exactly as in EDGE, so the value depends on the
  frame rate: the same motion at 50 fps scores (30/50)^2 = 0.36x its 30 fps value. Only compare
  PFC numbers computed at the same fps."""
  DT = 1.0 / fps
  flat = [i for i in range(3) if i != up]
  root_v = (root_pos[1:] - root_pos[:-1]) / DT
  root_a = (root_v[1:] - root_v[:-1]) / DT
  root_a[:, up] = np.maximum(root_a[:, up], 0)
  root_a = np.linalg.norm(root_a, axis=-1)
  root_a = root_a / root_a.max()
  feet = np.stack([l_ankle, l_toe, r_ankle, r_toe], 1)
  foot_v = np.linalg.norm(feet[2:, :, flat] - feet[1:-1, :, flat], axis=-1)
  left = np.minimum(foot_v[:, 0], foot_v[:, 1])
  right = np.minimum(foot_v[:, 2], foot_v[:, 3])
  return float((left * right * root_a).mean() * 10000)


def body_point(motion, body, offset):
  """World position of a point fixed in a body (offset in the body frame)."""
  q = motion["body_quat"][:, body]
  return motion["body_pos"][:, body] + quat_apply(q, np.broadcast_to(offset, q.shape[:-1] + (3,)))


# ------------------------------------------------------------------ beats and lag


def kinetic_velocity(body_pos):
  """Bailando calc_db: mean over bodies of the per-frame displacement norm. Length T-1, unit m/frame."""
  return np.mean(np.linalg.norm(body_pos[1:] - body_pos[:-1], axis=-1), axis=1)


def kinematic_beats(body_pos, fps, smooth_sigma_s):
  """Local minima (scipy argrelextrema, np.less, order 1) of the Gaussian-smoothed kinetic velocity.
  Smoothing sigma = smooth_sigma_s * fps frames (Bailando: 5 frames at 60 fps)."""
  kv = kinetic_velocity(body_pos)
  sigma = smooth_sigma_s * fps
  kvs = gaussian_filter1d(kv, sigma) if sigma > 0 else kv
  return argrelextrema(kvs, np.less)[0], kvs


def beat_alignment(kin_beats, music_beats, sigma_frames):
  """Returns both directions of the AIST++ Beat Alignment Score:
    'kin_to_music'  (primary; AIST++/FACT paper definition and this project's spec): mean over
                    kinematic beats of exp(-d^2 / (2 sigma^2)), d = distance to the nearest music beat
    'music_to_kin'  (Bailando utils/beat_align_score.py BA(), used by EDGE's paper): mean over music
                    beats of the same kernel to the nearest kinematic beat
  Distances in frames; sigma in frames."""
  kb, mb = np.asarray(kin_beats), np.asarray(music_beats)
  out = {"n_kinematic_beats": int(kb.size), "n_music_beats": int(mb.size)}
  if kb.size == 0 or mb.size == 0:
    out.update(kin_to_music=0.0, music_to_kin=0.0)
    return out
  d1 = np.min(np.abs(kb[:, None] - mb[None, :]), axis=1).astype(float)
  d2 = np.min(np.abs(mb[:, None] - kb[None, :]), axis=1).astype(float)
  out["kin_to_music"] = float(np.mean(np.exp(-d1 ** 2 / (2 * sigma_frames ** 2))))
  out["music_to_kin"] = float(np.mean(np.exp(-d2 ** 2 / (2 * sigma_frames ** 2))))
  return out


def beat_alignment_chance(kin_beats, music_beats, sigma_frames, n=500, seed=0):
  """Chance level of the kinematic->music score for this pair: the music beat grid is shifted by a
  uniform random offset in [0, mean beat spacing) n times (seeded) and rescored. Keeps the tempo and
  the beat count, destroys the phase. Returns mean, 5th and 95th percentile. The mean is the chance
  level; the percentiles only describe the spread over phases (offsets near 0 or near one spacing are
  nearly the original alignment, so p95 is close to the aligned score for a periodic motion)."""
  mb = np.asarray(music_beats, float)
  if len(mb) < 2 or len(kin_beats) == 0:
    return {"mean": None, "p5": None, "p95": None, "n": 0}
  per = float(np.mean(np.diff(mb)))
  rng = np.random.default_rng(seed)
  sc = [beat_alignment(kin_beats, mb + rng.uniform(0, per), sigma_frames)["kin_to_music"] for _ in range(n)]
  return {"mean": float(np.mean(sc)), "p5": float(np.percentile(sc, 5)),
          "p95": float(np.percentile(sc, 95)), "n": n, "beat_spacing_frames": per}


def synthetic_beat_motion(music_beats, n_frames, n_bodies=30, amp=0.2, seed=0):
  """Positive control: bodies oscillate as amp*cos(pi*phi(t)), with phi = beat index interpolated in time,
  so every body's speed is exactly zero on each music beat (and only there, up to phi's kinks)."""
  mb = np.asarray(music_beats, float)
  per = np.mean(np.diff(mb))
  knots_t = np.concatenate([[mb[0] - per * 50], mb, [mb[-1] + per * 50]])
  knots_p = np.concatenate([[-50.0], np.arange(len(mb), dtype=float), [len(mb) - 1 + 50.0]])
  phi = np.interp(np.arange(n_frames, dtype=float), knots_t, knots_p)
  rng = np.random.default_rng(seed)
  base = rng.normal(0, 0.3, (n_bodies, 3))
  dirs = rng.normal(0, 1, (n_bodies, 3))
  dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
  return base[None] + amp * np.cos(np.pi * phi)[:, None, None] * dirs[None]


def lag_frames(ref_curve, exe_curve, max_lag):
  """Shift (frames) that maximises the normalised cross-correlation of two curves on the same grid.
  Positive = the executed curve happens later than the reference. Each curve is z-scored over the
  overlap at each shift (Pearson correlation), so the score does not favour short overlaps.
  Returns (integer lag, correlation at it, parabolic sub-frame estimate)."""
  a, b = np.asarray(ref_curve, float), np.asarray(exe_curve, float)
  n = min(len(a), len(b))
  a, b = a[:n], b[:n]
  lags = np.arange(-max_lag, max_lag + 1)
  r = np.full(len(lags), -np.inf)
  for i, L in enumerate(lags):
    if L >= 0:
      x, y = a[:n - L], b[L:]
    else:
      x, y = a[-L:], b[:n + L]
    if len(x) < 10 or x.std() == 0 or y.std() == 0:
      continue
    r[i] = float(np.corrcoef(x, y)[0, 1])
  k = int(np.argmax(r))
  sub = float(lags[k])
  if 0 < k < len(lags) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
    den = r[k - 1] - 2 * r[k] + r[k + 1]
    if den != 0:
      sub = float(lags[k] + 0.5 * (r[k - 1] - r[k + 1]) / den)
  return int(lags[k]), float(r[k]), sub


# ------------------------------------------------------------------ smoothness / limits


def joint_jerk(joint_pos, fps):
  """|d^3 q / dt^3| per frame and joint (third forward difference / dt^3), rad/s^3."""
  if len(joint_pos) < 4:
    return np.full((0, joint_pos.shape[1]), np.nan)
  return np.abs(np.diff(joint_pos, n=3, axis=0)) * fps ** 3


def near_limit_fraction(x, lo, hi, margin):
  """Per-column fraction of frames with x within `margin` of [lo, hi] (or outside it)."""
  near = (x <= lo + margin) | (x >= hi - margin)
  return near.mean(axis=0)
