"""Convert EDGE's SMPL motion output to an AMASS-style SMPL-X .npz for GMR.

EDGE writes (with `--save_motions`) a pickle per dance:
    smpl_poses : (T, 72)  axis-angle, 24 SMPL joints (root already rotated to Z-up)
    smpl_trans : (T, 3)   root (pelvis) position in metres, Z-up
    full_pose  : (T, 24, 3) FK joint positions (unused here)
at 30 fps with a neutral, betas=0 body.

GMR (General Motion Retargeting) loads SMPL-X .npz files in the AMASS SMPL-X
layout and reads: gender, betas, root_orient, pose_body, trans,
mocap_frame_rate. This script writes those plus the rest of the AMASS keys
(poses, pose_hand, pose_jaw, pose_eye, surface_model_type) for compatibility.

Mapping:
  * root_orient = SMPL joint 0. Both are Z-up, so no frame change is needed.
  * pose_body   = SMPL joints 1..21. SMPL-X's 21 body joints use the same
    order. SMPL's hand joints 22/23 have no body slot in SMPL-X and are dropped,
    and the hands stay at their rest (flat) pose.
  * floor       = EDGE's floor plane is z = 1 m (see vis.py), so z is shifted
    down by --floor_z (default 1.0) to put the floor at z = 0 for GMR.
  * trans       = pelvis - J_pelvis. EDGE's "trans" is the pelvis position,
    but a SMPL-X model puts the pelvis at J_pelvis(betas) + trans. J_pelvis
    comes from the SMPL-X model file if you pass one (--smplx_model).
    Otherwise the script uses the neutral SMPL-X rest pelvis.

Note: SMPL and SMPL-X have slightly different rest skeletons. Copying the joint
rotations over is the standard approximation and is accurate enough for
retargeting.

Usage:
  python g1/edge_to_smplx.py motions/ -o smplx_out/           # a folder of .pkl
  python g1/edge_to_smplx.py dance.pkl -o dance_smplx.npz      # a single file
  python g1/edge_to_smplx.py motions/ -o out/ --smplx_model /path/to/SMPLX_NEUTRAL.npz

Then, in GMR:
  python scripts/smplx_to_robot.py --smplx_file out/dance.npz --robot unitree_g1 ...
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

EDGE_FPS = 30.0
NUM_BETAS = 16
# Rest-pose pelvis joint of the neutral SMPL-X model (betas = 0), in the model's
# own canonical frame. Used only when no --smplx_model is given.
SMPLX_NEUTRAL_PELVIS = np.array([0.00312, -0.35140, 0.01203])
EDGE_FLOOR_Z = 1.0  # vis.py draws EDGE's ground plane at z = 1


def pelvis_from_model(model_path: Path, betas: np.ndarray) -> np.ndarray:
  """Rest-pose pelvis = J_regressor[0] @ (v_template + shapedirs @ betas)."""
  if model_path.suffix == ".npz":
    m = dict(np.load(model_path, allow_pickle=True))
  else:  # .pkl (may hold a scipy sparse J_regressor)
    with open(model_path, "rb") as f:
      m = pickle.load(f, encoding="latin1")
  jreg = m["J_regressor"]
  jreg = jreg.toarray() if hasattr(jreg, "toarray") else np.asarray(jreg)
  verts = np.asarray(m["v_template"], dtype=np.float64)
  shapedirs = np.asarray(m["shapedirs"], dtype=np.float64)
  n = min(len(betas), shapedirs.shape[-1])
  verts = verts + shapedirs[..., :n] @ betas[:n]
  return jreg[0] @ verts


def convert(pkl_path: Path, out_path: Path, pelvis_rest: np.ndarray, gender: str, floor_z: float = EDGE_FLOOR_Z) -> None:
  with open(pkl_path, "rb") as f:
    data = pickle.load(f)

  poses = np.asarray(data["smpl_poses"], dtype=np.float64).reshape(-1, 24, 3)
  pelvis = np.asarray(data["smpl_trans"], dtype=np.float64).reshape(-1, 3).copy()
  pelvis[:, 2] -= floor_z
  T = poses.shape[0]
  assert pelvis.shape[0] == T, f"{pkl_path}: {T} pose frames vs {pelvis.shape[0]} trans frames"

  root_orient = poses[:, 0]                      # (T, 3)
  pose_body = poses[:, 1:22].reshape(T, 63)      # (T, 63)
  pose_jaw = np.zeros((T, 3))
  pose_eye = np.zeros((T, 6))
  pose_hand = np.zeros((T, 90))                  # 15 joints x 2 hands, flat
  trans = pelvis - pelvis_rest[None]

  # AMASS SMPL-X "poses" = root(3) body(63) jaw(3) eyes(6) hands(90) = 165
  full = np.concatenate([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], axis=1)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out_path,
    gender=np.array(gender),
    surface_model_type=np.array("smplx"),
    mocap_frame_rate=np.array(EDGE_FPS),
    betas=np.zeros(NUM_BETAS),
    root_orient=root_orient.astype(np.float32),
    pose_body=pose_body.astype(np.float32),
    pose_hand=pose_hand.astype(np.float32),
    pose_jaw=pose_jaw.astype(np.float32),
    pose_eye=pose_eye.astype(np.float32),
    poses=full.astype(np.float32),
    trans=trans.astype(np.float32),
  )
  print(f"{pkl_path.name} -> {out_path}  ({T} frames, {T / EDGE_FPS:.1f}s)")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("input", type=Path, help="EDGE motion .pkl or a folder of them")
  p.add_argument("-o", "--output", type=Path, required=True, help="output .npz (single input) or folder")
  p.add_argument("--smplx_model", type=Path, default=None,
                 help="SMPLX_NEUTRAL.npz/.pkl, to compute the exact rest pelvis offset")
  p.add_argument("--floor_z", type=float, default=EDGE_FLOOR_Z,
                 help="height of EDGE's floor, subtracted from z (default 1.0; 0 keeps raw EDGE heights)")
  p.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
  args = p.parse_args()

  if args.smplx_model is not None:
    pelvis_rest = pelvis_from_model(args.smplx_model, np.zeros(NUM_BETAS))
    print(f"SMPL-X rest pelvis from {args.smplx_model.name}: {np.round(pelvis_rest, 5)}")
  else:
    pelvis_rest = SMPLX_NEUTRAL_PELVIS
    print(f"Using built-in neutral SMPL-X rest pelvis {pelvis_rest} (pass --smplx_model to compute it)")

  if args.input.is_dir():
    files = sorted(args.input.glob("*.pkl"))
    if not files:
      raise SystemExit(f"No .pkl files in {args.input}")
    for f in files:
      convert(f, args.output / f"{f.stem}.npz", pelvis_rest, args.gender, args.floor_z)
  else:
    out = args.output if args.output.suffix == ".npz" else args.output / f"{args.input.stem}.npz"
    convert(args.input, out, pelvis_rest, args.gender, args.floor_z)


if __name__ == "__main__":
  main()
