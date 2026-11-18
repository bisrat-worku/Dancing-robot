"""Constrained-RL variant of mjlab's G1 tracking task: a leg-error constraint with a learned multiplier.

Registers  Mjlab-Tracking-Flat-Unitree-G1-LegConstraint  next to the stock task, which is left untouched
(its config is rebuilt from mjlab's own factory and only a new reward term is appended).

Reward     every stock term unchanged, plus
             r_leg = lambda * exp(-e_leg^2 / sigma^2)        (weight 1.0, dt-scaled like every stock term)
           e_leg   mean over the 6 leg bodies (left/right hip_roll_link, knee_link, ankle_roll_link) of the
                   Euclidean distance |body_pos_relative_w - robot_body_pos_w|, i.e. in the same frame as the
                   stock motion_body_pos term (reference re-anchored to the robot's root position and heading).
                   Mean of per-body distances, as mjlab's error_body_pos metric and policy_eval's group errors.
           sigma   the stock motion_body_pos std (read from the stock config, 0.3 m)
           lambda = 0 makes the term exactly 0.0, so the total equals the stock reward bit for bit
           (checked by g1/leg_constraint_check.py equal).
Multiplier once per PPO iteration, after the rollout and before the policy update:
             e_batch = mean of e_leg over every env and step of the rollout
             e_bar  <- e_batch on the first batch, else 0.9 * e_bar + 0.1 * e_batch
             lambda <- clip(lambda + 0.02 * (e_bar - eps) / eps, 0, 2),  lambda starts at 0
           eps = 0.9 * the stock run's leg error at iteration 20000 (EPS below; source in EPS_SOURCE).
           lambda and e_bar are saved in every checkpoint (infos["leg_constraint"]) and restored on resume;
           a stock checkpoint has neither, so training from it starts at lambda = 0 with e_bar unset.
Logging    every iteration, to TensorBoard (Constraint/*) and to <log_dir>/leg_constraint.csv.

Import this module (from the mjlab venv) to register the task; g1/train_leg_constraint.py does that and
then runs mjlab's train CLI.
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import _REGISTRY, register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp import MotionCommand

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-LegConstraint"
TERM = "leg_constraint"
LEG_BODIES = ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link")
ARM_BODIES = ("left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
              "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
TORSO_BODIES = ("pelvis", "torso_link")

# eps = 0.9 * e_leg of the stock run at iteration 20000. The stock training logs have no leg-group error
# (only Metrics/motion/error_body_pos over all 14 bodies), so it was measured from the stock final checkpoint
# under the training sampling: g1/leg_constraint_check.py eps (see EPS_SOURCE).
# Measured 2026-11-18 on run 2026-11-17_20-06-34_funkorama/model_19998.pt (the stock 10k->20k continuation's
# final checkpoint, i.e. "iteration 20000"): 4096 envs, seed 42, 50 warm-up + 100 measured iterations of 24 steps,
# stochastic actions. SD over iterations 0.027 cm. The all-body error from the same sums (3.427 cm) matches the
# logged error_body_pos of that rollout (3.426 cm) and the end of the stock training log (3.42 cm).
STOCK_E_LEG_20K = 0.047189
EPS = round(0.9 * STOCK_E_LEG_20K, 6)  # 0.04247 m
EPS_SOURCE = ("computed from the stock checkpoint model_19998.pt under the training sampling "
              "(g1/leg_constraint_check.py eps); the training logs have no leg-group error")
EMA = 0.9
STEP = 0.02
LAMBDA_MAX = 2.0


def group_error(command: "MotionCommand", names) -> torch.Tensor:
  """Mean over the named bodies of |p_ref,relative - p_robot| (m), per env."""
  idx = [i for i, n in enumerate(command.cfg.body_names) if n in names]
  assert len(idx) == len(names), (names, command.cfg.body_names)
  d = command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx]
  return torch.norm(d, dim=-1).mean(-1)


class ConstraintState:
  """Multiplier, its EMA, and the sums over the current rollout. Lives on the env as env.leg_constraint."""

  def __init__(self):
    self.lam = 0.0
    self.e_bar: float | None = None
    self.reset_batch()

  def reset_batch(self):
    self.sums: dict[str, torch.Tensor | float] = {}
    self.n = 0

  def add(self, key, x: torch.Tensor):
    v = x.detach().sum()
    self.sums[key] = self.sums.get(key, 0.0) + v

  def means(self):
    return {k: float(v) / self.n for k, v in self.sums.items()} if self.n else {}


def state_of(env) -> ConstraintState:
  env = getattr(env, "unwrapped", env)
  if not hasattr(env, "leg_constraint"):
    env.leg_constraint = ConstraintState()
  return env.leg_constraint


def leg_constraint_reward(env: "ManagerBasedRlEnv", command_name: str, std: float, body_names: tuple,
                          eps: float, ema: float, step: float, lambda_max: float) -> torch.Tensor:
  """lambda * exp(-e_leg^2 / std^2). eps/ema/step/lambda_max are the multiplier's settings, used by
  LegConstraintRunner; they are params here so they are recorded in the run's params/env.yaml."""
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  st = state_of(env)
  e_leg = group_error(command, body_names)
  # rollout sums for the multiplier and the per-iteration log
  st.add("e_leg", e_leg)
  st.add("e_arms", group_error(command, ARM_BODIES))
  st.add("e_torso", group_error(command, TORSO_BODIES))
  st.add("e_body", group_error(command, tuple(command.cfg.body_names)))
  st.n += e_leg.numel()
  r = st.lam * torch.exp(-e_leg.square() / std**2)
  st.add("r_leg", r)
  return r


def make_env_cfg(play: bool = False):
  if EPS is None:
    raise RuntimeError("leg_constraint.EPS is not set; measure it first: g1/leg_constraint_check.py eps")
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  stock = cfg.rewards["motion_body_pos"]
  cfg.rewards[TERM] = RewardTermCfg(
    func=leg_constraint_reward, weight=1.0,
    params={"command_name": "motion", "std": float(stock.params["std"]), "body_names": LEG_BODIES,
            "eps": float(EPS), "ema": EMA, "step": STEP, "lambda_max": LAMBDA_MAX})
  return cfg


class LegConstraintRunner(MotionTrackingOnPolicyRunner):
  """Stock tracking runner plus the once-per-iteration multiplier update, its logging and checkpointing."""

  def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
    super().__init__(env, train_cfg, log_dir, device, registry_name)
    self.st = state_of(self.env)
    p = self.env.unwrapped.cfg.rewards[TERM].params
    self.eps, self.ema, self.step_size, self.lam_max = p["eps"], p["ema"], p["step"], p["lambda_max"]
    self._last = None
    self._csv = None

    step = self.env.step

    def step_and_sum(actions):  # batch mean of the total reward (and of the stock part)
      obs, rew, dones, extras = step(actions)
      self.st.add("reward", rew)
      self.st.add("n_done", dones.float())
      return obs, rew, dones, extras

    self.env.step = step_and_sum

    update = self.alg.update

    def constrained_update(*a, **k):
      self._constraint_update()
      return update(*a, **k)

    self.alg.update = constrained_update
    log = self.logger.log

    def log_with_constraint(*a, **k):
      log(*a, **k)
      self._log(k["it"] if "it" in k else a[0])

    self.logger.log = log_with_constraint

  def _constraint_update(self):
    st = self.st
    m = st.means()
    lam_used = st.lam
    e_batch = m["e_leg"]
    st.e_bar = e_batch if st.e_bar is None else self.ema * st.e_bar + (1 - self.ema) * e_batch
    st.lam = min(max(st.lam + self.step_size * (st.e_bar - self.eps) / self.eps, 0.0), self.lam_max)
    # every mean is over (envs x rollout steps); reward is the env's dt-scaled total, r_leg is raw
    self._last = {"lambda": st.lam, "lambda_used": lam_used, "e_bar": st.e_bar, "e_batch": e_batch,
                  "eps": self.eps, "e_arms": m["e_arms"], "e_torso": m["e_torso"], "e_body": m["e_body"],
                  "reward_per_step": m["reward"],
                  "r_leg_per_step": m["r_leg"] * self.env.unwrapped.step_dt,
                  "dones_per_step": m["n_done"]}
    self._last["stock_reward_per_step"] = self._last["reward_per_step"] - self._last["r_leg_per_step"]
    st.reset_batch()

  def _log(self, it):
    if self._last is None or self.logger.writer is None:
      return
    for k, v in self._last.items():
      self.logger.writer.add_scalar(f"Constraint/{k}", v, it)
    if self._csv is None:
      path = os.path.join(self.logger.log_dir, "leg_constraint.csv")
      new = not os.path.exists(path)
      self._csv_f = open(path, "a", newline="")
      self._csv = csv.writer(self._csv_f)
      if new:
        self._csv.writerow(["iteration", *self._last])
    self._csv.writerow([it, *(f"{v:.6g}" for v in self._last.values())])
    self._csv_f.flush()
    L = self._last
    print(f"[leg_constraint] it {it}  lambda {L['lambda']:.4f}  e_bar {100 * L['e_bar']:.3f} cm  "
          f"e_batch {100 * L['e_batch']:.3f} cm  eps {100 * L['eps']:.3f} cm  arms {100 * L['e_arms']:.3f}  "
          f"torso {100 * L['e_torso']:.3f}  reward/step {L['reward_per_step']:.4f}", flush=True)

  def save(self, path, infos=None):
    infos = {**(infos or {}), "leg_constraint": {"lambda": self.st.lam, "e_bar": self.st.e_bar}}
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    s = (infos or {}).get("leg_constraint")
    if s is None:
      self.st.lam, self.st.e_bar = 0.0, None
      print(f"[leg_constraint] {os.path.basename(path)} has no multiplier state (stock checkpoint): "
            "lambda = 0, e_bar set from the first batch")
    else:
      self.st.lam, self.st.e_bar = float(s["lambda"]), s["e_bar"]
      print(f"[leg_constraint] restored lambda = {self.st.lam:.4f}, e_bar = {self.st.e_bar} from {path}")
    return infos


def register():
  if TASK_ID in _REGISTRY:
    return
  register_mjlab_task(task_id=TASK_ID, env_cfg=make_env_cfg(), play_env_cfg=make_env_cfg(play=True),
                      rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=LegConstraintRunner)


if EPS is not None:
  register()
