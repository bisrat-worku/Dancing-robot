"""Convert a G1 retarget (GMR format) into an mjlab motion-tracking reference (motion.npz).

Input:  <run>/<song>_g1_fixed.pkl  (falls back to <song>_g1.pkl), as written by edge_to_g1.py /
        fix_g1_feet.py:  fps, root_pos (T,3), root_rot (T,4 xyzw), dof_pos (T,29).
Output (next to the input):
  <song>_mjlab.csv          mjlab's CSV interchange format: one row per frame,
                            root_pos(3), root_quat(4, xyzw), dof_pos(29), no header.
  <song>_mjlab_motion.npz   the reference mjlab's tracking task reads
                            (--env.commands.motion.motion-file):
                            fps, joint_pos/joint_vel (T,29), body_pos_w/body_quat_w(wxyz)/
                            body_lin_vel_w/body_ang_vel_w (T, nbody, 3|4), at --output_fps.

It does what mjlab's scripts/csv_to_npz.py does (resample 30 -> 50 Hz with lerp/slerp,
finite-difference velocities, replay through mjlab's own G1 to record every body's world
state), but saves locally and only uploads to W&B with --upload, instead of always uploading.
GMR's and mjlab's G1 have the same 29 joints in the same order and identical body FK
(checked: 0 mm difference), so no remapping is needed.

Optional --smooth_hz applies a zero-phase low-pass (Butterworth, filtfilt) to root and joints
before resampling. EDGE dances have joint-speed spikes (~30 rad/s), and the tracking reward
also scores body velocities, so spikes in the reference become spikes in the policy.

Usage (any python; it re-launches itself in mjlab's venv):
  python3 g1/edge_to_mjlab.py outputs/<run>/                    # -> <song>_mjlab_motion.npz
  python3 g1/edge_to_mjlab.py outputs/<run>/ --smooth_hz 0      # no smoothing
  python3 g1/edge_to_mjlab.py outputs/<run>/ --upload funkorama # also push to W&B registry 'motions'
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

MJLAB_DIR = Path(os.environ.get("MJLAB_DIR", "/home/bwork/mjlab"))
MJLAB_PY = MJLAB_DIR / ".venv" / "bin" / "python"

try:
  import mjlab  # noqa: F401
except ImportError:
  if not MJLAB_PY.exists():
    sys.exit(f"mjlab venv not found at {MJLAB_PY}. Run: cd {MJLAB_DIR} && uv sync --extra cu130")
  if Path(sys.executable).resolve() == MJLAB_PY.resolve():
    raise
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402

from mjlab.scene import Scene  # noqa: E402
from mjlab.scripts.csv_to_npz import MotionLoader  # noqa: E402
from mjlab.sim.sim import Simulation, SimulationCfg  # noqa: E402
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg  # noqa: E402

# mjlab's joint order for the G1 (scripts/csv_to_npz.py), identical to GMR's g1_mocap_29dof.xml.
G1_JOINTS = [
  f"{s}_{j}_joint"
  for s in ("left", "right")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
  f"{s}_{j}_joint"
  for s in ("left", "right")
  for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")
]


def find_input(path: Path, which: str | None) -> Path:
  if path.is_file():
    return path
  for suffix in ([which] if which else ["g1_fixed", "g1"]):
    hits = sorted(path.glob(f"*_{suffix}.pkl"))
    if hits:
      if suffix == "g1" and not which:
        print("[warn] no _g1_fixed.pkl; using raw _g1.pkl (feet may go below the floor). "
              "Run fix_g1_feet.py first for a cleaner reference.")
      return hits[0]
  sys.exit(f"No *_g1_fixed.pkl or *_g1.pkl in {path}. Run edge_to_g1.py / fix_g1_feet.py first.")


def lowpass(x: np.ndarray, fps: float, hz: float) -> np.ndarray:
  b, a = butter(4, hz / (fps / 2))
  return filtfilt(b, a, x, axis=0, padtype="odd", padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def smooth(root_pos, root_xyzw, dof, fps, hz):
  q = root_xyzw.copy()
  for t in range(1, len(q)):  # make the quaternion sign continuous before filtering
    if np.dot(q[t], q[t - 1]) < 0:
      q[t] = -q[t]
  q = lowpass(q, fps, hz)
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  return lowpass(root_pos, fps, hz), q, lowpass(dof, fps, hz)


def foot_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, geoms: list[int]) -> float:
  """Lowest world z over the foot collision geoms (capsules/spheres/boxes)."""
  low = np.inf
  for g in geoms:
    c, rmat, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
      z = c[2] - size[0]
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      z = c[2] - abs(rmat[2, 2]) * size[1] - size[0]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
      z = c[2] - np.abs(rmat[2, :]) @ size
    else:
      z = c[2]
    low = min(low, z)
  return low


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("input", type=Path, help="outputs/<run>/ folder, or a *_g1*.pkl")
  ap.add_argument("--which", choices=["g1_fixed", "g1"], help="pick the input explicitly (default: g1_fixed, else g1)")
  ap.add_argument("--smooth_hz", type=float, default=8.0, help="low-pass cutoff in Hz, 0 = off (default 8)")
  ap.add_argument("--output_fps", type=float, default=50.0, help="mjlab policy rate (default 50)")
  ap.add_argument("-o", "--out", type=Path, help="output .npz (default <song>_mjlab_motion.npz)")
  ap.add_argument("--upload", metavar="NAME", help="also log to the W&B registry 'motions' as NAME")
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  a = ap.parse_args()

  src = find_input(a.input, a.which)
  song = src.stem.split("_g1")[0]
  out_npz = a.out or src.with_name(f"{song}_mjlab_motion.npz")
  out_csv = src.with_name(f"{song}_mjlab.csv")

  with open(src, "rb") as f:
    g1 = pickle.load(f)
  fps = float(g1["fps"])
  root_pos = np.asarray(g1["root_pos"], np.float64)
  root_q = np.asarray(g1["root_rot"], np.float64)  # xyzw
  dof = np.asarray(g1["dof_pos"], np.float64)
  assert dof.shape[1] == 29, f"expected 29 G1 dofs, got {dof.shape}"
  print(f"[in]  {src}  ({len(dof)} frames @ {fps:g} fps = {len(dof) / fps:.1f} s)")

  raw_speed = np.abs(np.diff(dof, axis=0)).max() * fps
  if a.smooth_hz > 0:
    root_pos, root_q, dof = smooth(root_pos, root_q, dof, fps, a.smooth_hz)
  sm_speed = np.abs(np.diff(dof, axis=0)).max() * fps

  # Scene + sim exactly as the tracking task builds them.
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=a.device)
  model = scene.compile()
  lo, hi = (np.array([model.joint(f"robot/{j}").range[k] for j in G1_JOINTS]) for k in (0, 1))
  n_clip = int(((dof < lo) | (dof > hi)).any(1).sum())
  dof = np.clip(dof, lo, hi)

  np.savetxt(out_csv, np.concatenate([root_pos, root_q, dof], 1), delimiter=",", fmt="%.7f")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / a.output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=a.device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  jidx = robot.find_joints(G1_JOINTS, preserve_order=True)[0]

  motion = MotionLoader(str(out_csv), fps, a.output_fps, a.device)
  T = motion.output_frames
  log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
  scene.reset()
  for _ in range(T):
    (bp, br, blv, bav, dp, dv), _ = motion.get_next_state()
    rs = robot.data.default_root_state.clone()
    rs[:, 0:3] = bp
    rs[:, :2] += scene.env_origins[:, :2]
    rs[:, 3:7], rs[:, 7:10], rs[:, 10:] = br, blv, bav
    robot.write_root_state_to_sim(rs)
    jp, jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    jp[:, jidx], jv[:, jidx] = dp, dv
    robot.write_joint_state_to_sim(jp, jv)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    d = robot.data
    for k, v in (("joint_pos", d.joint_pos), ("joint_vel", d.joint_vel), ("body_pos_w", d.body_link_pos_w),
                 ("body_quat_w", d.body_link_quat_w), ("body_lin_vel_w", d.body_link_lin_vel_w),
                 ("body_ang_vel_w", d.body_link_ang_vel_w)):
      log[k].append(v[0].cpu().numpy().copy())
  arrays = {k: np.stack(v) for k, v in log.items()}
  np.savez(out_npz, fps=[a.output_fps], **arrays)

  # Floor check on mjlab's own foot collision geometry (what the policy will actually touch).
  mm = sim.mj_model
  md = mujoco.MjData(mm)
  feet = [g for g in range(mm.ngeom) if "foot" in mm.geom(g).name and "collision" in mm.geom(g).name]
  csv = np.loadtxt(out_csv, delimiter=",")
  lows = []
  for row in csv:
    md.qpos[:3], md.qpos[3:7], md.qpos[7:] = row[:3], row[[6, 3, 4, 5]], row[7:]
    mujoco.mj_kinematics(mm, md)
    lows.append(foot_bottom_z(mm, md, feet))
  lows = np.array(lows)

  print(f"[out] {out_csv}")
  print(f"[out] {out_npz}  ({T} frames @ {a.output_fps:g} fps, {arrays['body_pos_w'].shape[1]} bodies)")
  print(f"      smoothing: {'off' if a.smooth_hz <= 0 else f'{a.smooth_hz:g} Hz'};  "
        f"peak joint speed {raw_speed:.1f} -> {sm_speed:.1f} rad/s;  frames clipped to joint limits: {n_clip}")
  print(f"      lowest foot point (mjlab capsules): min {lows.min() * 100:+.1f} cm, "
        f"5th pct {np.percentile(lows, 5) * 100:+.1f} cm, median {np.median(lows) * 100:+.1f} cm")
  print(f"      pelvis z {arrays['body_pos_w'][:, 0, 2].min():.2f}-{arrays['body_pos_w'][:, 0, 2].max():.2f} m")
  if lows.min() < -0.02:
    print("[warn] feet go >2 cm below the floor in the reference; the policy can't follow that.")

  if a.upload:
    import wandb

    run = wandb.init(project="csv_to_npz", name=a.upload)
    art = wandb.Artifact(name=a.upload, type="motions")
    art.add_file(str(out_npz), name="motion.npz")  # mjlab's train/play look for exactly this name
    art = run.log_artifact(art)
    run.link_artifact(artifact=art, target_path=f"wandb-registry-motions/{a.upload}")
    run.finish()
    print(f"[W&B] uploaded; train with --registry-name <your-org>/wandb-registry-motions/{a.upload}")


if __name__ == "__main__":
  main()
