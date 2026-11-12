"""Quality report for a retargeted EDGE -> Unitree G1 dance.

  python3 g1/eval_g1_motion.py outputs/<run>/                  # re-launches itself in GMR's venv
  python3 g1/eval_g1_motion.py outputs/<run>/ --g1 g1_fixed    # evaluate fix_g1_feet.py's output
Writes <run>/<song>_<g1>_metrics.json and prints a report.

Checks (all on the saved <song>_g1.pkl, replayed through MuJoCo forward kinematics):
  1. Fidelity    G1 body pose vs GMR's scaled human targets (the exact poses the IK aimed at):
                 position error (cm) and orientation error (deg) per matched body, mean / max / when.
                 Also the error after removing each body's constant link-origin offset.
  2. Floor       lowest point of the real foot geometry (4 collision spheres per foot: 2 heel, 2 toe,
                 and the foot mesh) vs z = 0: max penetration depth, fraction of frames below.
  3. Contacts    EDGE's own predicted contacts (pkl "contact", > 0.95 = planted, EDGE's threshold):
                 G1 heel/toe height (floating) and horizontal speed (skating) while planted.
                 EDGE's own joint speed in those frames is reported too, to see who skates.
  4. Limits      joint-range violations and fraction of frames within MARGIN of a limit.
  5. Jumps       joint velocity vs the URDF velocity limits, acceleration, and frame-to-frame
                 outliers (robust z-score); outliers are checked against EDGE's 2.5 s slice seams.
  6. Root        height range, root linear / angular speed, sudden root jumps.
"""

import json
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GMR_DIR = Path(os.environ.get("GMR_DIR", "/home/bwork/GMR"))
GMR_PY = GMR_DIR / ".venv" / "bin" / "python"
try:
  import general_motion_retargeting  # noqa: F401
except ImportError:
  if Path(sys.executable).resolve() == GMR_PY.resolve():
    raise
  os.execv(str(GMR_PY), [str(GMR_PY), os.path.abspath(__file__), *sys.argv[1:]])

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_to_g1 as e2g  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402

FPS = 30
CONTACT_THRESH = 0.95          # EDGE's static-foot threshold (model/diffusion.py)
LIMIT_MARGIN = np.radians(3)   # "near a limit" = within 3 degrees
OUTLIER_Z = 8.0                # robust z-score (median/MAD) for frame-to-frame jumps
SEAM_PERIOD = 75               # EDGE stitches 5 s slices every 2.5 s = 75 frames
FEET = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
URDF = GMR_DIR / "assets" / "unitree_g1" / "g1_custom_collision_29dof.urdf"


def t_of(i):
  return f"{i / FPS:.2f}s (frame {i})"


def stats(x):
  x = np.asarray(x, dtype=float)
  if x.size == 0:
    return {"n": 0}
  return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max()),
          "argmax": int(np.argmax(x))}


def robust_outliers(x, z=OUTLIER_Z):
  """Indices where x is > z robust standard deviations above its median."""
  med = np.median(x)
  mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
  return np.where((x - med) / mad > z)[0]


def main():
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("run", type=Path)
  ap.add_argument("--g1", default="g1", help="which G1 file: g1 = <song>_g1.pkl, g1_fixed = <song>_g1_fixed.pkl")
  args = ap.parse_args()
  run = args.run
  edge_pkl = [f for f in sorted(run.glob("*.pkl")) if "_g1" not in f.stem][0]
  song = edge_pkl.stem
  with open(edge_pkl, "rb") as f:
    edge = e2g._NumpyCompatUnpickler(f).load()
  frames, _ = e2g.edge_to_frames(edge_pkl, e2g.EDGE_FLOOR_Z)
  with open(run / f"{song}_{args.g1}.pkl", "rb") as f:
    g1 = pickle.load(f)
  T = len(frames)
  qpos = np.concatenate([g1["root_pos"], g1["root_rot"][:, [3, 0, 1, 2]], g1["dof_pos"]], axis=1)

  gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot="unitree_g1",
                                 actual_human_height=e2g.DEFAULT_HEIGHT, verbose=False)
  model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
  data = mj.MjData(model)
  bid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)  # noqa: E731

  pairs = {}  # human joint -> robot body (table2 is the final IK stage; same bodies as table1)
  for tbl in (gmr.ik_match_table1, gmr.ik_match_table2):
    for body, (hj, wp, wr, _, _) in tbl.items():
      pairs[hj] = (body, wp, wr)
  names = list(pairs)
  body_ids = [bid(pairs[n][0]) for n in names]

  # foot geometry: collision spheres (heel = x<0, toe = x>0 in the foot frame) + visual mesh
  foot_geoms = {}
  for side, body in FEET.items():
    b = bid(body)
    g_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
    spheres = [g for g in g_ids if model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE]
    meshes = [g for g in g_ids if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
    foot_geoms[side] = {
      "heel": [g for g in spheres if model.geom_pos[g][0] < 0],
      "toe": [g for g in spheres if model.geom_pos[g][0] > 0],
      "mesh": meshes,
    }

  def mesh_verts(g):
    m = model.geom_dataid[g]
    a, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
    return model.mesh_vert[a:a + n]

  # ---------------- replay ----------------
  pos_err = np.zeros((T, len(names)))
  rot_err = np.zeros((T, len(names)))
  local_off = np.zeros((T, len(names), 3))
  low = {s: {k: np.zeros(T) for k in ("heel", "toe", "mesh")} for s in FEET}
  pt_xy = {s: {k: np.zeros((T, 2)) for k in ("heel", "toe")} for s in FEET}
  for t in range(T):
    data.qpos[:] = qpos[t]
    mj.mj_kinematics(model, data)
    gmr.update_targets({k: (v[0].copy(), v[1].copy()) for k, v in frames[t].items()})
    tgt = gmr.scaled_human_data
    for j, (n, b) in enumerate(zip(names, body_ids)):
      tp, tq = tgt[n]
      d = data.xpos[b] - tp
      pos_err[t, j] = np.linalg.norm(d)
      rt = R.from_quat(tq, scalar_first=True)
      rot_err[t, j] = np.degrees((R.from_quat(data.xquat[b], scalar_first=True).inv() * rt).magnitude())
      local_off[t, j] = rt.inv().apply(d)
    for s, gs in foot_geoms.items():
      for k in ("heel", "toe"):
        zs = [data.geom_xpos[g][2] - model.geom_size[g][0] for g in gs[k]]
        low[s][k][t] = min(zs)
        pt_xy[s][k][t] = np.mean([data.geom_xpos[g][:2] for g in gs[k]], axis=0)
      low[s]["mesh"][t] = min(
        (data.geom_xpos[g] + mesh_verts(g) @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()
        for g in gs["mesh"])

  report = {"run": str(run), "g1_file": f"{song}_{args.g1}.pkl", "frames": T, "seconds": T / FPS}
  lines = []
  P = lines.append

  # 1. fidelity
  resid = np.linalg.norm(local_off - local_off.mean(0), axis=2)
  fid = {}
  P("== 1. Fidelity vs GMR's scaled targets ==")
  P(f"{'human':15}{'G1 body':25}{'pos mean':>9}{'pos max':>9}  {'at':18}{'rot mean':>9}{'rot max':>8}  "
    f"{'at':18}{'offset':>7}{'resid':>7}{'resid max':>10}")
  for j, n in enumerate(names):
    body, wp, wr = pairs[n]
    fid[n] = {"robot_body": body, "pos_cm": stats(pos_err[:, j] * 100), "rot_deg": stats(rot_err[:, j]),
              "const_offset_cm": float(np.linalg.norm(local_off[:, j].mean(0)) * 100),
              "resid_cm": stats(resid[:, j] * 100), "pos_weight": wp, "rot_weight": wr}
    f = fid[n]
    P(f"{n:15}{body:25}{f['pos_cm']['mean']:9.2f}{f['pos_cm']['max']:9.2f}  {t_of(f['pos_cm']['argmax']):18}"
      f"{f['rot_deg']['mean']:9.1f}{f['rot_deg']['max']:8.1f}  {t_of(f['rot_deg']['argmax']):18}"
      f"{f['const_offset_cm']:7.1f}{f['resid_cm']['mean']:7.2f}{f['resid_cm']['max']:10.2f}"
      + ("  (orientation-only task)" if wp == 0 else ""))
  tracked = np.array([pairs[n][1] > 0 for n in names])
  P(f"position-tracked bodies: pos mean {pos_err[:, tracked].mean() * 100:.2f} cm, "
    f"residual mean {resid[:, tracked].mean() * 100:.2f} cm; all bodies rot mean {rot_err.mean():.1f} deg")
  report["fidelity"] = fid

  # 2. floor penetration
  P("\n== 2. Floor penetration (z = 0) ==")
  floor = {}
  for s in FEET:
    for k in ("heel", "toe", "mesh"):
      z = low[s][k]
      floor[f"{s}_{k}"] = {"min_z_cm": float(z.min() * 100), "at": int(z.argmin()),
                           "frac_below_0": float((z < 0).mean()), "frac_below_-1cm": float((z < -0.01).mean())}
      f = floor[f"{s}_{k}"]
      P(f"{s:5} {k:4}: lowest {f['min_z_cm']:6.2f} cm at {t_of(f['at']):18} below floor {f['frac_below_0'] * 100:5.1f}% "
        f"of frames (deeper than 1 cm: {f['frac_below_-1cm'] * 100:4.1f}%)")
  lowest = np.minimum.reduce([low[s][k] for s in FEET for k in ("heel", "toe", "mesh")])
  P(f"whole robot feet: max depth {max(0, -lowest.min()) * 100:.2f} cm, below floor in {(lowest < 0).mean() * 100:.1f}% of frames; "
    f"lowest foot point median {np.median(lowest) * 100:.2f} cm")
  report["floor"] = floor

  # 3. floating & skating while EDGE says planted
  P("\n== 3. Floating / skating while EDGE's contact flag > 0.95 ==")
  cont = {}
  if "contact" in edge:
    c = np.asarray(edge["contact"])[:T] > CONTACT_THRESH  # l_ankle, r_ankle, l_toe, r_toe
    edge_j = {"left_heel": 7, "right_heel": 8, "left_toe": 10, "right_toe": 11}
    chan = {"left_heel": 0, "right_heel": 1, "left_toe": 2, "right_toe": 3}
    hum = np.asarray(edge["full_pose"])[:T]
    for key, ch in chan.items():
      s, k = key.split("_")
      planted = c[:-1, ch]
      h = low[s][k][:-1][planted]
      v = np.linalg.norm(np.diff(pt_xy[s][k], axis=0), axis=1)[planted] * FPS
      vh = np.linalg.norm(np.diff(hum[:, edge_j[key], :2], axis=0), axis=1)[planted] * FPS
      cont[key] = {"planted_frac": float(planted.mean()), "height_cm": stats(h * 100),
                   "g1_speed_cm_s": stats(v * 100), "edge_speed_cm_s": stats(vh * 100),
                   "frac_floating_gt_2cm": float((h > 0.02).mean()) if h.size else None,
                   "frac_skating_gt_10cm_s": float((v > 0.10).mean()) if v.size else None}
      f = cont[key]
      if h.size:
        P(f"{key:11} planted {f['planted_frac'] * 100:5.1f}% | height mean {f['height_cm']['mean']:5.2f} "
          f"p95 {f['height_cm']['p95']:5.2f} max {f['height_cm']['max']:5.2f} cm, >2 cm {f['frac_floating_gt_2cm'] * 100:4.1f}% | "
          f"G1 slide mean {f['g1_speed_cm_s']['mean']:5.1f} p95 {f['g1_speed_cm_s']['p95']:5.1f} max "
          f"{f['g1_speed_cm_s']['max']:5.1f} cm/s, >10 cm/s {f['frac_skating_gt_10cm_s'] * 100:4.1f}% | "
          f"EDGE slide mean {f['edge_speed_cm_s']['mean']:5.1f} p95 {f['edge_speed_cm_s']['p95']:5.1f} cm/s")
      else:
        P(f"{key:11} never planted")
  else:
    P("no 'contact' in the EDGE pkl (generated before contacts were saved); skipped")
  report["contacts"] = cont

  # 4. joint limits
  P("\n== 4. Joint limits ==")
  jn = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
  lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
  q = g1["dof_pos"]
  viol = (q < lo - 1e-4) | (q > hi + 1e-4)
  near = (q - lo < LIMIT_MARGIN) | (hi - q < LIMIT_MARGIN)
  P(f"violations: {int(viol.sum())} joint-frames; frames with any joint within 3 deg of a limit: "
    f"{near.any(1).mean() * 100:.1f}%")
  per = sorted(((jn[j], near[:, j].mean()) for j in range(len(jn))), key=lambda x: -x[1])
  P("most often near a limit: " + ", ".join(f"{n} {f * 100:.1f}%" for n, f in per[:6] if f > 0))
  report["limits"] = {"violations": int(viol.sum()), "frac_frames_near_any": float(near.any(1).mean()),
                      "near_frac_per_joint": {n: float(f) for n, f in per}}

  # 5. joint velocity / acceleration / jumps
  P("\n== 5. Joint velocity, acceleration and jumps ==")
  vlim = {}
  for j in ET.parse(URDF).getroot().iter("joint"):
    lim = j.find("limit")
    if lim is not None and lim.get("velocity"):
      vlim[j.get("name")] = float(lim.get("velocity"))
  vel = np.diff(q, axis=0) * FPS
  acc = np.diff(q, n=2, axis=0) * FPS ** 2
  rows, over_total = [], 0
  for j, n in enumerate(jn):
    v = np.abs(vel[:, j])
    lim = vlim.get(n, np.nan)
    over = int((v > lim).sum())
    over_total += over
    rows.append((n, float(v.max()), int(v.argmax()), lim, over, float(np.abs(acc[:, j]).max())))
  rows.sort(key=lambda r: -(r[1] / r[3]))
  P(f"{'joint':28}{'max |v| rad/s':>14}{'limit':>7}{'ratio':>7}{'frames over':>12}  {'at':18}{'max |a| rad/s^2':>16}")
  for n, vm, at, lim, over, am in rows[:8]:
    P(f"{n:28}{vm:14.1f}{lim:7.0f}{vm / lim:7.2f}{over:12d}  {t_of(at):18}{am:16.0f}")
  P(f"joint-frames over the URDF velocity limit: {over_total} (of {vel.size}); "
    f"frames with any joint over: {int(((np.abs(vel) > np.array([vlim.get(n, np.inf) for n in jn])).any(1)).sum())}")
  step = np.abs(np.diff(q, axis=0)).max(1)  # biggest joint change per frame
  out = robust_outliers(step)
  seam = [i for i in out if min((i + 1) % SEAM_PERIOD, SEAM_PERIOD - (i + 1) % SEAM_PERIOD) <= 2]
  P(f"frame-to-frame outliers (> {OUTLIER_Z:g} robust SD of the max joint step): {len(out)}"
    + (f" at {[round(float(i) / FPS, 2) for i in out[:12]]} s" if len(out) else "")
    + f"; on EDGE slice seams (+-2 frames): {len(seam)}")
  report["joints"] = {"velocity": [dict(zip(("joint", "max", "argmax", "limit", "frames_over", "max_acc"), r))
                                   for r in rows],
                      "frames_over_limit_total": over_total,
                      "outlier_frames": [int(i) for i in out], "outliers_on_seams": [int(i) for i in seam]}

  # 6. root
  P("\n== 6. Root ==")
  rp = g1["root_pos"]
  rq = R.from_quat(g1["root_rot"])  # xyzw
  lin = np.linalg.norm(np.diff(rp, axis=0), axis=1) * FPS
  ang = np.degrees((rq[:-1].inv() * rq[1:]).magnitude()) * FPS
  rjump = robust_outliers(lin)
  ajump = robust_outliers(ang)
  P(f"height {rp[:, 2].min():.3f} .. {rp[:, 2].max():.3f} m (mean {rp[:, 2].mean():.3f}); "
    f"horizontal travel {np.linalg.norm(rp[-1, :2] - rp[0, :2]):.2f} m")
  P(f"linear speed max {lin.max():.2f} m/s at {t_of(int(lin.argmax()))}, p95 {np.percentile(lin, 95):.2f}; "
    f"angular speed max {ang.max():.0f} deg/s at {t_of(int(ang.argmax()))}, p95 {np.percentile(ang, 95):.0f}")
  P(f"sudden root jumps (> {OUTLIER_Z:g} robust SD): linear {len(rjump)}"
    + (f" at {[round(float(i) / FPS, 2) for i in rjump[:10]]} s" if len(rjump) else "")
    + f", angular {len(ajump)}" + (f" at {[round(float(i) / FPS, 2) for i in ajump[:10]]} s" if len(ajump) else ""))
  report["root"] = {"z_min": float(rp[:, 2].min()), "z_max": float(rp[:, 2].max()),
                    "lin_speed_max": float(lin.max()), "ang_speed_max_deg": float(ang.max()),
                    "lin_jumps": [int(i) for i in rjump], "ang_jumps": [int(i) for i in ajump]}

  print("\n".join(lines))
  out_json = run / f"{song}_{args.g1}_metrics.json"
  out_json.write_text(json.dumps(report, indent=1))
  print(f"\nmetrics -> {out_json}")


if __name__ == "__main__":
  main()
