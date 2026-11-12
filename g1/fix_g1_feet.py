"""Post-process a G1 retarget: no floor penetration + contact-aware foot locking.

  python3 g1/fix_g1_feet.py outputs/<run>/            # writes <song>_g1_fixed.pkl (re-launches in GMR's venv)
  python3 g1/fix_g1_feet.py outputs/<run>/ --video    # + <song>_g1_fixed.mp4

Inputs: <song>.pkl (EDGE, needs "contact", saved since 2026-11-07) and <song>_g1.pkl (edge_to_g1.py).
The original <song>_g1.pkl is left untouched.

1. Contacts   per foot planted = EDGE heel or toe flag > 0.95, cleaned: gaps <= GAP_FILL frames are filled,
              plants shorter than MIN_PLANT frames are dropped (raw flags flicker, median plant = 3 frames).
2. Targets    foot body (ankle_roll_link) keeps its original orientation, except that it is levelled
              (pitch/roll -> 0, yaw kept) while EDGE says heel AND toe are planted; its position is corrected:
                planted: xy pinned to the plant's mean xy (shift capped at MAX_SHIFT), and z set so the lowest
                         point of the real foot geometry (collision spheres + mesh) touches z = 0
                free:    z raised only if the foot would go below z = 0
              xy corrections fade in/out over BLEND frames around each plant; z changes are smoothed but the
              "never below the floor" bound is applied after smoothing, so it is always respected.
3. Leg IK     each leg's 6 joints (hip p/r/y, knee, ankle p/r) are solved with damped least squares in
              MuJoCo to the corrected 6-D foot pose, warm-started from the original pose, joint limits
              clamped. Where a leg can't reach (knee straight) the pelvis is lowered by the shortfall
              (smoothed) and the IK re-run, up to DROP_PASSES times. Pelvis xy/rotation, waist and
              arm joints are not changed.
"""

import argparse
import os
import pickle
import sys
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
from scipy.ndimage import gaussian_filter1d, maximum_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_to_g1 as e2g  # noqa: E402
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402

FPS = 30
CONTACT_THRESH = 0.95
GAP_FILL = 3       # frames
MIN_PLANT = 5      # frames
BLEND = 5          # frames to fade xy corrections in / out
MAX_SHIFT = 0.12   # m, cap on the horizontal pin correction
Z_SMOOTH = 1.5     # frames (gaussian sigma) for the z correction
FLAT_SMOOTH = 2.0  # frames (gaussian sigma) for blending the foot to flat
IK_ITERS = 50
DROP_PASSES = 4    # IK passes that lower the pelvis where a leg can't reach its foot target
DROP_SMOOTH = 3    # frames
MAX_DROP = 0.08    # m, cap on how far the pelvis may be lowered
IK_DAMP = 1e-3
ROT_W = 0.3        # m per rad: weight of foot orientation vs position in the IK
FEET = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
LEG_JOINTS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
CHANNELS = {"left": (0, 2), "right": (1, 3)}  # EDGE contact: l_ankle, r_ankle, l_toe, r_toe


def segments(mask):
  e = np.diff(np.r_[0, mask.astype(int), 0])
  return list(zip(np.where(e == 1)[0], np.where(e == -1)[0]))  # [start, end)


def clean_contacts(mask):
  m = mask.copy()
  segs = segments(m)
  for (_, e0), (s1, _) in zip(segs[:-1], segs[1:]):
    if s1 - e0 <= GAP_FILL:
      m[e0:s1] = True
  for s, e in segments(m):
    if e - s < MIN_PLANT:
      m[s:e] = False
  return m


def foot_local_points(model, body):
  """Points (body frame) whose min z is the foot's lowest point: sphere bottoms are handled by radius."""
  pts, rad = [], []
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != body:
      continue
    gp, gr = model.geom_pos[g], R.from_quat(model.geom_quat[g], scalar_first=True)
    if model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE:
      pts.append(gp[None]), rad.append([model.geom_size[g][0]])
    elif model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
      m = model.geom_dataid[g]
      v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
      pts.append(gp + gr.apply(v)), rad.append(np.zeros(len(v)))
  return np.concatenate(pts), np.concatenate(rad)


def lowest(p, rot, local, rad):
  """Lowest world z of the foot for body positions p (T,3), rotations rot (T)."""
  z = np.einsum("tij,nj->tni", rot.as_matrix(), local)[..., 2] + p[:, None, 2] - rad[None]
  return z.min(1)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run", type=Path, help="outputs/<run>/ folder")
  ap.add_argument("--video", action="store_true", help="also render <song>_g1_fixed.mp4")
  a = ap.parse_args()

  edge_pkl = [f for f in sorted(a.run.glob("*.pkl")) if "_g1" not in f.stem][0]
  song = edge_pkl.stem
  with open(edge_pkl, "rb") as f:
    edge = e2g._NumpyCompatUnpickler(f).load()
  if "contact" not in edge:
    sys.exit(f"{edge_pkl} has no 'contact' (generated before 2026-11-07); regenerate it with generate_dance.sh")
  with open(a.run / f"{song}_g1.pkl", "rb") as f:
    g1 = pickle.load(f)
  qpos = np.concatenate([g1["root_pos"], g1["root_rot"][:, [3, 0, 1, 2]], g1["dof_pos"]], axis=1)
  T = len(qpos)
  contact = np.asarray(edge["contact"])[:T] > CONTACT_THRESH

  model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
  data = mj.MjData(model)
  bid = {s: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for s, b in FEET.items()}

  # original foot poses
  p0 = {s: np.zeros((T, 3)) for s in FEET}
  q0 = {s: np.zeros((T, 4)) for s in FEET}
  for t in range(T):
    data.qpos[:] = qpos[t]
    mj.mj_kinematics(model, data)
    for s, b in bid.items():
      p0[s][t], q0[s][t] = data.xpos[b], data.xquat[b]

  # ---- corrected foot targets ----
  target, planted_all, flat_all = {}, {}, {}
  for s, b in bid.items():
    rot = R.from_quat(q0[s], scalar_first=True)
    local, rad = foot_local_points(model, b)
    raw = contact[:, CHANNELS[s][0]] | contact[:, CHANNELS[s][1]]
    planted = clean_contacts(raw)
    planted_all[s] = planted

    # xy pin: inside a plant, shift to the plant's mean xy; fade the edge offsets out over BLEND frames
    dxy = np.zeros((T, 2))
    wsum = np.zeros(T)
    for st, en in segments(planted):
      anchor = p0[s][st:en, :2].mean(0)
      d = anchor - p0[s][st:en, :2]
      n = np.linalg.norm(d, axis=1, keepdims=True)
      d *= np.minimum(1, MAX_SHIFT / np.maximum(n, 1e-9))
      dxy[st:en] = d
      wsum[st:en] = 1
      for k in range(1, BLEND + 1):
        w = 0.5 * (1 + np.cos(np.pi * k / (BLEND + 1)))  # smooth 1 -> 0
        for t, edge_d in ((st - k, d[0]), (en - 1 + k, d[-1])):
          if 0 <= t < T and not planted[t] and w > wsum[t]:  # strongest nearby fade wins
            dxy[t], wsum[t] = w * edge_d, w

    # flat foot when EDGE says heel AND toe are planted: keep yaw, remove pitch/roll (smoothly blended)
    both = clean_contacts(contact[:, CHANNELS[s][0]]) & clean_contacts(contact[:, CHANNELS[s][1]]) & planted
    f = np.clip(gaussian_filter1d(both.astype(float), FLAT_SMOOTH), 0, 1)
    f[both] = 1.0
    yaw = R.from_euler("z", rot.as_euler("ZYX")[:, 0])
    rv = (rot.inv() * yaw).as_rotvec()  # rotation from original to flat, scaled by f
    rot = rot * R.from_rotvec(rv * f[:, None])
    flat_all[s] = both

    low0 = lowest(p0[s], rot, local, rad)
    dz_raw = np.where(planted, -low0, np.maximum(0, -low0))
    dz = np.maximum(gaussian_filter1d(dz_raw, Z_SMOOTH), -low0)  # never below the floor
    target[s] = (p0[s] + np.c_[dxy, dz], rot.as_quat(scalar_first=True))

  # ---- leg IK ----
  dof_adr = {s: [model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint")]
                 for j in LEG_JOINTS] for s in FEET}
  qadr = {s: [model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint")]
              for j in LEG_JOINTS] for s in FEET}
  jrange = {s: model.jnt_range[[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint")
                                for j in LEG_JOINTS]] for s in FEET}
  jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

  def solve(drop):
    """Leg IK for every frame with the pelvis lowered by drop (T,). Returns qpos, residuals."""
    out = qpos.copy()
    out[:, 2] -= drop
    res = {s: np.zeros((T, 4)) for s in FEET}  # pos err xyz (target - achieved, m), rot err (rad)
    for t in range(T):
      data.qpos[:] = out[t]
      for s, b in bid.items():
        tp, tq = target[s][0][t], R.from_quat(target[s][1][t], scalar_first=True)
        for _ in range(IK_ITERS):
          mj.mj_kinematics(model, data)
          mj.mj_comPos(model, data)
          ep = tp - data.xpos[b]
          er = (tq * R.from_quat(data.xquat[b], scalar_first=True).inv()).as_rotvec()
          if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
          mj.mj_jacBody(model, data, jacp, jacr, b)
          J = np.vstack([jacp[:, dof_adr[s]], ROT_W * jacr[:, dof_adr[s]]])
          e = np.r_[ep, ROT_W * er]
          dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(6), e)
          data.qpos[qadr[s]] = np.clip(data.qpos[qadr[s]] + dq, jrange[s][:, 0], jrange[s][:, 1])
        mj.mj_kinematics(model, data)
        res[s][t] = np.r_[tp - data.xpos[b],
                          (tq * R.from_quat(data.xquat[b], scalar_first=True).inv()).magnitude()]
      out[t] = data.qpos
    return out, res

  # a foot target the leg can't reach (knee already straight) -> lower the pelvis by the shortfall,
  # smoothed over time (max-filter + gaussian so the envelope covers each miss), and solve again
  drop = np.zeros(T)
  for it in range(DROP_PASSES + 1):
    out, res = solve(drop)
    short = np.max([-res[s][:, 2] for s in FEET], axis=0)  # foot stuck above its target (m)
    if it == DROP_PASSES or short.max() < 0.003:
      break
    need = np.where(short > 0.002, short, 0)
    drop = np.minimum(drop + gaussian_filter1d(maximum_filter1d(need, 2 * DROP_SMOOTH + 1), DROP_SMOOTH),
                      MAX_DROP)
  resid = {s: np.c_[np.linalg.norm(res[s][:, :3], axis=1), res[s][:, 3]] for s in FEET}

  dst = a.run / f"{song}_g1_fixed.pkl"
  with open(dst, "wb") as f:
    pickle.dump({**g1, "root_pos": out[:, :3], "dof_pos": out[:, 7:]}, f)

  for s in FEET:
    planted = planted_all[s]
    shift = np.linalg.norm(target[s][0] - p0[s], axis=1)
    print(f"{s:5}: planted {planted.mean() * 100:4.1f}% in {len(segments(planted))} plants | foot moved mean "
          f"{shift.mean() * 100:.2f} max {shift.max() * 100:.2f} cm | IK residual pos mean "
          f"{resid[s][:, 0].mean() * 1000:.2f} max {resid[s][:, 0].max() * 1000:.1f} mm, rot max "
          f"{np.degrees(resid[s][:, 1].max()):.2f} deg")
    bad = np.where(resid[s][:, 0] > 0.01)[0]
    print(f"       flat {flat_all[s].mean() * 100:4.1f}% | IK misses > 1 cm: {len(bad)} frames"
          + (f" at {sorted(set(round(float(i) / FPS, 1) for i in bad))[:15]} s" if len(bad) else ""))
  print(f"pelvis lowered: mean {drop.mean() * 100:.2f} max {drop.max() * 100:.2f} cm "
        f"(in {(drop > 0.002).mean() * 100:.1f}% of frames)")
  print(f"saved {dst}")
  if a.video:
    vpath = dst.with_suffix(".mp4")
    e2g.render_video(out, "unitree_g1", str(vpath), FPS)
    print(f"video {vpath}")


if __name__ == "__main__":
  main()
