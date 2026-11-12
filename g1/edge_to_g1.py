"""Retarget an EDGE dance straight to the Unitree G1 with GMR, no SMPL-X body model needed.

GMR's retargeter (GeneralMotionRetargeting.retarget) only consumes, per frame,
    {joint_name: (world_position_xyz, world_quat_wxyz)}
for the SMPL-X body joints named in ik_configs/smplx_to_g1.json (pelvis, spine3,
hips, knees, feet, shoulders, elbows, wrists). GMR's SMPL-X loader builds that dict by
running the SMPL-X model. EDGE's .pkl already holds the same information from EDGE's
own SMPL skeleton:
  full_pose  (T, 24, 3)  world joint positions (EDGE's FK)
  smpl_poses (T, 72)     local axis-angle rotations, root already Z-up
World orientations are chained down the SMPL tree exactly as GMR's get_smplx_data does
(R_world[i] = R_world[parent] * R_local[i]). SMPL and SMPL-X share the names and order of
the first 22 joints. EDGE's floor is at z = 1 m (vis.py), so z is shifted down by --floor_z.

Output (<song>_g1.pkl) is in the format of GMR's smplx_to_robot.py --save_path:
  fps, root_pos (T,3), root_rot (T,4 xyzw), dof_pos (T,29), local_body_pos=None, link_body_list=None
so GMR's scripts/vis_robot_motion.py can replay it.

Usage (any Python; it re-launches itself in GMR's venv):
  python g1/edge_to_g1.py outputs/funkorama_20261106_225909/                # folder or .pkl
  python g1/edge_to_g1.py dance.pkl -o dance_g1.pkl --video dance_g1.mp4     # + offscreen G1 render
  python g1/edge_to_g1.py dance.pkl --view                                   # + interactive viewer
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
  if not GMR_PY.exists():
    sys.exit(f"GMR venv not found at {GMR_PY}. Run: cd {GMR_DIR} && uv sync --python 3.10")
  if Path(sys.executable).resolve() == GMR_PY.resolve():
    raise
  os.execv(str(GMR_PY), [str(GMR_PY), os.path.abspath(__file__), *sys.argv[1:]])

# --- MuJoCo GPU rendering on WSL (GALLIUM d3d12 routes GL to the GPU; headless EGL cannot) ---
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from tqdm import tqdm  # noqa: E402

from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402

EDGE_FPS = 30
EDGE_FLOOR_Z = 1.0
# GMR's estimate for a betas=0 SMPL-X body (utils/smpl.py: 1.66 + 0.1 * betas[0]).
# EDGE uses the mean SMPL body, so this is what GMR would use for EDGE's SMPL-X export too.
DEFAULT_HEIGHT = 1.66

# SMPL kinematic tree and the first 22 joint names, shared with SMPL-X (smplx.joint_names).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINT_NAMES = [
  "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
  "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
  "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
# EDGE's rest-pose bone offsets (vis.py smpl_offsets), used only for the consistency check.
SMPL_OFFSETS = np.array([
  [0.0, 0.0, 0.0], [0.05858135, -0.08228004, -0.01766408], [-0.06030973, -0.09051332, -0.01354254],
  [0.00443945, 0.12440352, -0.03838522], [0.04345142, -0.38646945, 0.008037],
  [-0.04325663, -0.38368791, -0.00484304], [0.00448844, 0.1379564, 0.02682033],
  [-0.01479032, -0.42687458, -0.037428], [0.01905555, -0.4200455, -0.03456167],
  [-0.00226458, 0.05603239, 0.00285505], [0.04105436, -0.06028581, 0.12204243],
  [-0.03483987, -0.06210566, 0.13032329], [-0.0133902, 0.21163553, -0.03346758],
  [0.07170245, 0.11399969, -0.01889817], [-0.08295366, 0.11247234, -0.02370739],
  [0.01011321, 0.08893734, 0.05040987], [0.12292141, 0.04520509, -0.019046],
  [-0.11322832, 0.04685326, -0.00847207], [0.2553319, -0.01564902, -0.02294649],
  [-0.26012748, -0.01436928, -0.03126873], [0.26570925, 0.01269811, -0.00737473],
  [-0.26910836, 0.00679372, -0.00602676],
])


class _NumpyCompatUnpickler(pickle.Unpickler):
  """EDGE's venv pickles with numpy 2.5, whose _frombuffer takes an extra axis_order
  argument that GMR's numpy 2.2 lacks. Accept it (None = plain C/F order)."""

  def find_class(self, module, name):
    if name == "_frombuffer" and module.startswith("numpy"):
      def _frombuffer(buf, dtype, shape, order, axis_order=None):  # copy of numpy 2.5's
        a = np.frombuffer(buf, dtype=dtype)
        if order == "K" and axis_order is not None:
          return a.reshape(shape, order="C").transpose(axis_order)
        return a.reshape(shape, order=order)
      return _frombuffer
    return super().find_class(module, name)


def edge_to_frames(pkl_path, floor_z):
  """EDGE .pkl -> list of GMR human_data dicts, one per frame."""
  with open(pkl_path, "rb") as f:
    d = _NumpyCompatUnpickler(f).load()
  local = np.asarray(d["smpl_poses"], dtype=np.float64).reshape(-1, 24, 3)[:, :22]
  pos = np.asarray(d["full_pose"], dtype=np.float64)[:, :22].copy()
  T = local.shape[0]
  pos[..., 2] -= floor_z

  world = [None] * 22
  for i in range(22):
    r = R.from_rotvec(local[:, i])
    world[i] = r if i == 0 else world[SMPL_PARENTS[i]] * r

  # Sanity check: re-run FK from these orientations; it must reproduce EDGE's positions.
  fk = np.zeros_like(pos)
  fk[:, 0] = pos[:, 0]
  for i in range(1, 22):
    p = SMPL_PARENTS[i]
    fk[:, i] = fk[:, p] + world[p].apply(SMPL_OFFSETS[i])
  err = np.abs(fk - pos).max()
  if err > 0.01:
    sys.exit(f"orientation/position mismatch {err:.3f} m: EDGE pkl doesn't match the SMPL tree")

  quats = np.stack([w.as_quat(scalar_first=True) for w in world], axis=1)  # (T,22,4) wxyz
  frames = [{n: (pos[t, j], quats[t, j]) for j, n in enumerate(JOINT_NAMES)} for t in range(T)]
  return frames, err


def render_video(qpos_all, robot, path, fps, width=960, height=720):
  import imageio
  import mujoco as mj
  from general_motion_retargeting.params import ROBOT_XML_DICT

  model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, height)
  data = mj.MjData(model)
  renderer = mj.Renderer(model, height=height, width=width)
  cam = mj.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 120.0
  with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
    for q in tqdm(qpos_all, desc="render"):
      data.qpos[:] = q
      mj.mj_forward(model, data)
      cam.lookat[:] = [q[0], q[1], 0.8]
      renderer.update_scene(data, camera=cam)
      w.append_data(renderer.render())
  renderer.close()


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("input", type=Path, help="EDGE motion .pkl, or an outputs/<run>/ folder")
  p.add_argument("-o", "--output", type=Path, help="output .pkl (default <input dir>/<song>_g1.pkl)")
  p.add_argument("--robot", default="unitree_g1")
  p.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="human height for GMR scaling")
  p.add_argument("--floor_z", type=float, default=EDGE_FLOOR_Z, help="EDGE floor height (default 1.0)")
  p.add_argument("--video", nargs="?", const="auto", help="render an .mp4 (default <output>.mp4)")
  p.add_argument("--view", action="store_true", help="replay in GMR's interactive viewer afterwards")
  a = p.parse_args()

  src = a.input
  if src.is_dir():
    cands = [f for f in sorted(src.glob("*.pkl")) if "_g1" not in f.stem]
    if len(cands) != 1:
      sys.exit(f"expected one EDGE .pkl in {src}, found {[c.name for c in cands]}")
    src = cands[0]
  out = a.output or src.with_name(f"{src.stem}_g1.pkl")

  frames, err = edge_to_frames(src, a.floor_z)
  print(f"{src.name}: {len(frames)} frames ({len(frames) / EDGE_FPS:.1f}s), FK check max err {err * 1000:.2f} mm")

  gmr = GeneralMotionRetargeting(
    src_human="smplx", tgt_robot=a.robot, actual_human_height=a.height, verbose=False
  )
  qpos = np.array([gmr.retarget(f).copy() for f in tqdm(frames, desc=f"retarget -> {a.robot}")])

  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "wb") as f:
    pickle.dump({
      "fps": EDGE_FPS,
      "root_pos": qpos[:, :3],
      "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # wxyz -> xyzw, as GMR saves it
      "dof_pos": qpos[:, 7:],
      "local_body_pos": None,
      "link_body_list": None,
    }, f)
  print(f"saved {out}  root_pos {qpos[:, :3].shape}, dof_pos {qpos[:, 7:].shape}")
  print(f"root height {qpos[:, 2].min():.2f}..{qpos[:, 2].max():.2f} m")

  if a.video:
    vpath = out.with_suffix(".mp4") if a.video == "auto" else Path(a.video)
    render_video(qpos, a.robot, str(vpath), EDGE_FPS)
    print(f"video {vpath}")

  if a.view:
    from general_motion_retargeting import RobotMotionViewer
    v = RobotMotionViewer(robot_type=a.robot, motion_fps=EDGE_FPS)
    try:
      while True:
        for q in qpos:
          v.step(root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:], rate_limit=True)
    except KeyboardInterrupt:
      pass
    v.close()


if __name__ == "__main__":
  main()
