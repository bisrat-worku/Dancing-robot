"""Roll out a trained mjlab G1 tracking policy (or the open-loop baseline) and save raw trajectories.

Runs in mjlab's venv (re-launches itself there). Called by evaluate.py; usable alone:
  python3 policy_eval/rollout.py verify   --run RUN_DIR
  python3 policy_eval/rollout.py clean    --run RUN_DIR --checkpoint CKPT --out traj.npz [--video V.mp4]
  python3 policy_eval/rollout.py random   --run RUN_DIR --checkpoint CKPT --out traj.npz --seeds 1000..1049
  python3 policy_eval/rollout.py openloop --run RUN_DIR --out traj.npz
  python3 policy_eval/rollout.py sweep    --run RUN_DIR --out traj.npz     (clean run on every checkpoint)
  python3 policy_eval/rollout.py push     --run RUN_DIR --checkpoint CKPT --out traj.npz --seeds ...
  python3 policy_eval/rollout.py friction --run RUN_DIR --checkpoint CKPT --out traj.npz --friction 0.3,0.6,1.2

Nothing here changes the training config, rewards, reference motion or checkpoints. The env is
built from mjlab's registry (the same code the run was trained with) and then only the evaluation
switches below are applied in memory:
  * 1 env, episode_length_s = 1e9 (no time_out), auto_reset = False (the terminal state is kept and
    nothing from a new episode can enter the record), sampling_mode = "start" (frame 0).
  * The motion command's end-of-clip wrap (which teleports the robot back to a resampled frame
    inside the same env.step) is replaced by a clamp at the last frame, so the last frame is real.
  * A passive feet/terrain contact sensor is added (reads contacts, applies no force).
  * clean / openloop / sweep / friction: no pushes, no startup DR, no reset perturbation, no
    observation noise. random: the training reset perturbation + startup DR (+ obs noise), no pushes.
    push: random + the training push event.
Frame alignment (from mjlab's env.step): after reset the robot is at reference frame 0; after env
step k (k >= 1) terminations are evaluated against reference frame k, and the recorded robot state is
the fully refreshed post-step state, so recorded frame k is compared with reference frame k.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from dataclasses import asdict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MOTOR_VELOCITY_LIMITS, EvalConfig  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.random import seed_rng  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FEET_SENSOR = "eval_feet_ground_contact"


# ---------------------------------------------------------------------------- run discovery


def find_training_run(run_dir: Path):
  """The training log folder with the most checkpoints, its checkpoints sorted by iteration."""
  runs = []
  for d in sorted((run_dir / "rl" / "g1_tracking").glob("*/")):
    ck = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ck:
      runs.append((d, ck))
  if not runs:
    raise SystemExit(f"no checkpoints under {run_dir}/rl/g1_tracking/")
  return runs


class _TaglessLoader(yaml.SafeLoader):
  """Reads mjlab's params/*.yaml without importing anything: python tags become plain data."""


def _tagless(loader, suffix, node):
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return loader.construct_scalar(node)


_TaglessLoader.add_multi_constructor("", _tagless)


def saved_env_cfg(train_dir: Path) -> dict:
  with open(train_dir / "params" / "env.yaml") as f:
    return yaml.load(f, Loader=_TaglessLoader)


# ---------------------------------------------------------------------------- env construction


def training_env_cfg(task, motion_file, num_envs, seed):
  cfg = load_env_cfg(task, play=False)
  cfg.commands["motion"].motion_file = str(motion_file)
  cfg.scene.num_envs = num_envs
  if cfg.scene.terrain is not None:  # mjlab copies scene.num_envs here when it builds the env
    cfg.scene.terrain.num_envs = num_envs
  cfg.seed = seed
  return cfg


def verify_training_cfg(task, train_dir: Path):
  """Rebuild the training env cfg from mjlab's registry and compare its YAML dump with the run's
  params/env.yaml (written by mjlab's train.py with the same dump). Returns the list of differences."""
  saved_txt = (train_dir / "params" / "env.yaml").read_text()
  saved = yaml.load(saved_txt, Loader=_TaglessLoader)
  cfg = training_env_cfg(task, saved["commands"]["motion"]["motion_file"],
                         int(saved["scene"]["num_envs"]), int(saved["seed"]))
  now_txt = yaml.dump(asdict(cfg), sort_keys=False)
  if now_txt == saved_txt:
    return []
  import difflib
  return [l for l in difflib.unified_diff(saved_txt.splitlines(), now_txt.splitlines(), lineterm="")
          if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]


def make_env(mode, motion_file, ecfg: EvalConfig, render=False, friction=None):
  cfg = training_env_cfg(ecfg.task, motion_file, 1, 0)
  cfg.episode_length_s = 1e9
  cfg.auto_reset = False
  mc = cfg.commands["motion"]
  mc.sampling_mode = "start"
  mc.debug_vis = render
  if mode in ("clean", "openloop", "sweep", "friction"):
    for k in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
      if k == "foot_friction" and mode == "friction":
        continue
      cfg.events.pop(k, None)
    if mode == "friction":
      cfg.events["foot_friction"].params["ranges"] = (float(friction), float(friction))
    mc.pose_range = {}
    mc.velocity_range = {}
    mc.joint_position_range = (0.0, 0.0)
    cfg.observations["actor"].enable_corruption = False
  elif mode in ("random", "push"):
    if mode == "random" and not ecfg.push_in_randomized:
      cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = bool(ecfg.random_obs_noise)
  else:
    raise ValueError(mode)
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (ContactSensorCfg(
    name=FEET_SENSOR,
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                         entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1),)
  if render:
    cfg.viewer.width, cfg.viewer.height = ecfg.video_width, ecfg.video_height
  env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode="rgb_array" if render else None)
  _clamp_motion_end(env)
  return env, cfg


def _clamp_motion_end(env):
  """MotionCommand._update_command without the end-of-clip wrap (see module docstring)."""
  cmd = env.command_manager.get_term("motion")
  T = cmd.motion.time_step_total

  def update(env_ids=None):
    if env_ids is None:
      cmd.time_steps += 1
    else:
      cmd.time_steps[env_ids] += 1
    cmd.time_steps.clamp_(max=T - 1)
    if cmd._pending_forward:
      cmd._pending_forward = False
      cmd._env.sim.forward()
    cmd.update_relative_body_poses()

  cmd._update_command = update


def load_policy(env_w, task, ckpt):
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env_w, asdict(agent_cfg), device=DEVICE)
  runner.load(str(ckpt), map_location=DEVICE)
  return runner.get_inference_policy(device=DEVICE)


# ---------------------------------------------------------------------------- model metadata


def model_meta(env):
  import mujoco
  robot = env.scene["robot"]
  m = env.sim.mj_model
  joint_names = list(robot.joint_names)
  body_names = list(robot.body_names)
  act_names = list(robot.actuator_names)
  ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
  forcerange = m.actuator_forcerange[ctrl_ids].copy()
  act_joint = [joint_names.index(a) for a in act_names]  # actuators are named after their joint
  all_j = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
  jids = []
  for n in joint_names:  # the scene model prefixes entity names ("robot/left_knee_joint")
    hits = [j for nm, j in all_j.items() if nm == n or (nm or "").endswith("/" + n)]
    assert len(hits) == 1, f"joint {n} matched {len(hits)} model joints"
    jids.append(hits[0])
  assert all(m.jnt_limited[j] for j in jids), "every actuated joint should have a range"
  jrange = m.jnt_range[jids].copy()
  assert (jrange[:, 1] > jrange[:, 0]).all()
  vel_lim = np.zeros(len(joint_names))
  vel_src = [""] * len(joint_names)
  for i, n in enumerate(joint_names):
    for pat, v, src in MOTOR_VELOCITY_LIMITS:
      if re.fullmatch(pat, n):
        vel_lim[i], vel_src[i] = v, src
        break
    else:
      raise RuntimeError(f"no motor velocity limit for joint {n}")
  # toe point of each foot: the front-most capsule end point of its sole collision geoms
  toe = {}
  for side in ("left", "right"):
    bids = [b for b in range(m.nbody) if re.fullmatch(
      rf"(?:.*/)?{side}_ankle_roll_link", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
    assert len(bids) == 1, f"{side}_ankle_roll_link not found once in the model"
    bid = bids[0]
    best = worst = None
    for g in range(m.ngeom):
      n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      if m.geom_bodyid[g] != bid or not re.fullmatch(rf"(?:.*/)?{side}_foot\d+_collision", n):
        continue
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, m.geom_quat[g])
      ax = R.reshape(3, 3)[:, 2]
      for s in (-1, 1):
        p = m.geom_pos[g] + s * m.geom_size[g, 1] * ax
        if best is None or p[0] > best[0]:
          best = p
        if worst is None or p[0] < worst[0]:
          worst = p
    assert best is not None, f"no {side} foot collision capsules found"
    toe[side] = best
    toe[side + "_heel"] = worst
  act = env.action_manager.get_term("joint_pos")
  scale = act.scale if isinstance(act.scale, float) else act.scale[0].cpu().numpy()
  offset = act.offset if isinstance(act.offset, float) else act.offset[0].cpu().numpy()
  return dict(joint_names=np.array(joint_names), body_names=np.array(body_names),
              actuator_names=np.array(act_names), actuator_joint=np.array(act_joint),
              actuator_forcerange=forcerange, joint_range=jrange, joint_vel_limit=vel_lim,
              joint_vel_limit_src=np.array(vel_src), toe_left=toe["left"], toe_right=toe["right"],
              heel_left=toe["left_heel"], heel_right=toe["right_heel"],
              action_scale=np.broadcast_to(scale, (len(joint_names),)).astype(np.float64),
              action_offset=np.broadcast_to(offset, (len(joint_names),)).astype(np.float64),
              step_dt=env.step_dt, physics_dt=env.physics_dt, decimation=env.cfg.decimation)


# ---------------------------------------------------------------------------- one episode


def run_episode(env, env_w, policy, *, seed, apply_startup_dr, openloop=False, frames=None,
                render_cb=None):
  """Seed -> (startup DR) -> reset at frame 0 -> step until the first termination or clip end."""
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("motion")
  T = cmd.motion.time_step_total
  n_steps = (T - 1) if frames is None else min(frames, T - 1)
  ids = torch.arange(env.num_envs, device=env.device)
  seed_rng(seed)
  if apply_startup_dr and "startup" in env.event_manager.available_modes:
    env.event_manager.apply(mode="startup", env_ids=ids)
  env.reset(env_ids=ids)
  obs = env_w.get_observations()
  assert int(cmd.time_steps[0]) == 1, "after reset the command should point at frame 1"

  B, J = robot.num_bodies, robot.num_joints
  rec = {k: torch.full((n_steps + 1,) + s, float("nan"), device=env.device) for k, s in {
    "body_pos": (B, 3), "body_quat": (B, 4), "body_lin_vel": (B, 3), "body_ang_vel": (B, 3),
    "joint_pos": (J,), "joint_vel": (J,), "action": (J,), "target": (J,),
    "torque": (env.cfg.decimation, robot.num_actuators), "contact_found": (2,),
    "contact_force": (2, 3), "mj_rel_pos": (len(cmd.cfg.body_names), 3),
    "mj_rel_quat": (len(cmd.cfg.body_names), 4)}.items()}
  rel_frame = torch.full((n_steps + 1,), -1, dtype=torch.long, device=env.device)
  term_names = list(env.termination_manager.active_terms)
  terms = torch.zeros((n_steps + 1, len(term_names)), dtype=torch.bool, device=env.device)
  ref_frame = torch.full((n_steps + 1,), -1, dtype=torch.long, device=env.device)
  sensor = env.scene.sensors[FEET_SENSOR]
  origin = env.scene.env_origins[0].clone()

  def snap(k):
    rec["body_pos"][k] = robot.data.body_link_pos_w[0] - origin
    rec["body_quat"][k] = robot.data.body_link_quat_w[0]
    rec["body_lin_vel"][k] = robot.data.body_link_lin_vel_w[0]
    rec["body_ang_vel"][k] = robot.data.body_link_ang_vel_w[0]
    rec["joint_pos"][k] = robot.data.joint_pos[0]
    rec["joint_vel"][k] = robot.data.joint_vel[0]
    rec["contact_found"][k] = (sensor.data.found[0].reshape(-1) > 0).float()
    rec["contact_force"][k] = sensor.data.force[0].reshape(-1, 3)

  sub = []
  orig_sim_step = env.sim.step

  def sim_step_logged():
    orig_sim_step()
    sub.append(robot.data.actuator_force[0].clone())

  env.sim.step = sim_step_logged
  snap(0)
  ref_frame[0] = 0
  if render_cb:
    render_cb(env)
  act_term = env.action_manager.get_term("joint_pos")
  first_term, last = None, n_steps
  try:
    for k in range(1, n_steps + 1):
      ref_frame[k] = cmd.time_steps[0]
      if openloop:
        q = cmd.motion.joint_pos[cmd.time_steps]
        a = (q - act_term.offset) / act_term.scale
      else:
        with torch.no_grad():
          a = policy(obs)
      sub.clear()
      obs, _, _, _ = env_w.step(a)
      rec["action"][k] = a[0]
      rec["target"][k] = act_term._processed_actions[0]
      rec["torque"][k] = torch.stack(sub)
      snap(k)
      # mjlab's anchor-relative reference as the env computed it after this step: reference frame
      # cmd.time_steps (= k + 1, clamped) re-anchored on the robot state of frame k
      rec["mj_rel_pos"][k] = cmd.body_pos_relative_w[0] - origin
      rec["mj_rel_quat"][k] = cmd.body_quat_relative_w[0]
      rel_frame[k] = cmd.time_steps[0]
      for i, n in enumerate(term_names):
        terms[k, i] = env.termination_manager.get_term(n)[0]
      if render_cb:
        render_cb(env)
      if bool(env.termination_manager.terminated[0]):
        first_term, last = k, k
        break
  finally:
    env.sim.step = orig_sim_step
  out = {k: v[: last + 1].cpu().numpy() for k, v in rec.items()}
  out["terminations"] = terms[: last + 1].cpu().numpy()
  out["ref_frame"] = ref_frame[: last + 1].cpu().numpy()
  out["rel_frame"] = rel_frame[: last + 1].cpu().numpy()
  out["tracked_body_names"] = list(cmd.cfg.body_names)
  out["first_term_frame"] = -1 if first_term is None else first_term
  out["term_names"] = term_names
  out["fired"] = [] if first_term is None else [n for i, n in enumerate(term_names)
                                                 if out["terminations"][first_term, i]]
  out["n_steps_clip"] = n_steps
  out["seed"] = seed
  if not (out["ref_frame"] == np.arange(last + 1)).all():
    raise RuntimeError("recorded frame k was not scored against reference frame k")
  return out


def save_episodes(path, episodes, meta, extra=None):
  arr = dict(meta)
  arr["n_episodes"] = len(episodes)
  for i, ep in enumerate(episodes):
    for k, v in ep.items():
      if k in ("term_names", "fired", "tracked_body_names"):
        v = np.array(v, dtype=str)
      arr[f"ep{i}/{k}"] = np.asarray(v)
  for k, v in (extra or {}).items():
    arr[k] = np.asarray(v)
  np.savez_compressed(path, **arr)


# ---------------------------------------------------------------------------- main


def parse_seeds(s):
  if ".." in s:
    a, b = s.split("..")
    return list(range(int(a), int(b) + 1))
  return [int(x) for x in s.split(",")]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["verify", "clean", "random", "openloop", "sweep", "push", "friction"])
  ap.add_argument("--run", type=Path, required=True)
  ap.add_argument("--train_dir", type=Path)
  ap.add_argument("--checkpoint", type=Path)
  ap.add_argument("--out", type=Path)
  ap.add_argument("--seeds", default="0")
  ap.add_argument("--friction", default="0.3,0.6,1.2")
  ap.add_argument("--video", type=Path)
  ap.add_argument("--frames", type=int, help="stop after this many steps (debug)")
  a = ap.parse_args()
  ecfg = EvalConfig()
  run = a.run.resolve()
  train_dir = a.train_dir or max(find_training_run(run), key=lambda r: len(r[1]))[0]
  saved = saved_env_cfg(train_dir)
  motion_file = saved["commands"]["motion"]["motion_file"]

  if a.mode == "verify":
    diffs = verify_training_cfg(ecfg.task, train_dir)
    print("CONFIG_MATCH" if not diffs else "CONFIG_DIFF\n" + "\n".join(diffs[:80]))
    return

  t0 = time.time()
  if a.mode == "sweep":
    ckpts = dict(find_training_run(run))[train_dir] if a.checkpoint is None else [a.checkpoint]
  else:
    ckpts = [a.checkpoint or dict(find_training_run(run))[train_dir][-1]]
  mode_env = {"sweep": "clean"}.get(a.mode, a.mode)
  frictions = [float(x) for x in a.friction.split(",")] if a.mode == "friction" else [None]
  episodes, extra = [], {"checkpoints": [], "frictions": []}
  meta = None
  for fr in frictions:
    env, _ = make_env(mode_env, motion_file, ecfg, render=a.video is not None, friction=fr)
    env_w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(ecfg.task).clip_actions)
    meta = meta or model_meta(env)
    writer = None
    if a.video:  # stream frames to the encoder: 1500 frames of 960x720 RGB do not fit in 7 GB of RAM
      import mediapy
      writer = mediapy.VideoWriter(str(a.video), shape=(ecfg.video_height, ecfg.video_width),
                                   fps=1.0 / env.step_dt)
      writer.__enter__()
    render_cb = (lambda e: writer.add_image(e.render())) if a.video else None
    for ck in ckpts:
      policy = None if a.mode == "openloop" else load_policy(env_w, ecfg.task, ck)
      seeds = parse_seeds(a.seeds) if a.mode in ("random", "push", "clean") else [0]
      for s in seeds:
        ep = run_episode(env, env_w, policy, seed=s, apply_startup_dr=a.mode in ("random", "push", "friction"),
                         openloop=a.mode == "openloop", frames=a.frames, render_cb=render_cb)
        ep["checkpoint"] = "" if policy is None else str(ck)
        episodes.append(ep)
        extra["checkpoints"].append(ep["checkpoint"])
        extra["frictions"].append(np.nan if fr is None else fr)
        status = "complete" if ep["first_term_frame"] < 0 else \
          f"terminated at frame {ep['first_term_frame']} by {','.join(ep['fired'])}"
        print(f"[{a.mode}] {Path(ep['checkpoint']).name if ep['checkpoint'] else 'open-loop'} "
              f"seed={s} friction={fr}: {status}  ({time.time() - t0:.0f} s)", flush=True)
    if writer is not None:
      writer.__exit__(None, None, None)
      print(f"video -> {a.video}")
    env.close()
    del env, env_w
    torch.cuda.empty_cache()
  extra["motion_file"] = motion_file
  extra["train_dir"] = str(train_dir)
  extra["mode"] = a.mode
  save_episodes(a.out, episodes, meta, extra)
  print(f"saved {len(episodes)} episode(s) -> {a.out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
