"""Checks for the leg-constraint variant (g1/leg_constraint.py). Re-launches itself in mjlab's venv.

  python3 g1/leg_constraint_check.py equal --train_dir <stock dir> --checkpoint model_9999.pt [--envs 1024 --steps 120]
      lambda = 0: the variant's total reward equals the stock reward on the same batch. A stock RewardManager
      (the stock task's reward config) is attached to the variant env and evaluated on the same state inside
      every step (two separate processes can't be compared: mjlab's GPU simulation does not reproduce
      bitwise across processes, two identical stock runs agree on only 2 % of rewards). Also checks that the
      variant's env config equals the stock one apart from the added term, and (lambda = 0.5) that the term
      adds exactly lambda * exp(-e_leg^2/sigma^2) * dt.
  python3 g1/leg_constraint_check.py eps --train_dir <stock dir> --checkpoint model_19998.pt [--warmup 50 --iters 100]
      e_leg of a stock checkpoint under the training sampling: the training env (4096 envs, seed 42, adaptive
      sampling, randomization, pushes, observation noise, random initial episode lengths), stochastic actions as
      in PPO's rollout, no policy update. e_leg is accumulated by the variant's reward term itself at lambda = 0
      (the equal check shows that this env is the stock env). The all-body error from the same sums is compared
      with the logged Metrics/motion/error_body_pos as a check of the procedure.
  python3 g1/leg_constraint_check.py shares --train_dir <stock dir> --checkpoint model_9999.pt --envs 512 --out F
      the stock reward's composition at a checkpoint under the training sampling: each term's per-step value and
      share of the total, and the motion_body_pos term's value at 3 and 5 cm of body-mean error.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

MJLAB_PY = Path(os.environ.get("MJLAB_DIR", "/home/bwork/mjlab")) / ".venv" / "bin" / "python"
try:
  import mjlab  # noqa: F401
except ImportError:
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leg_constraint as LC  # noqa: E402

if LC.EPS is None:  # the checks never update lambda, so any eps will do for building the env here
  LC.EPS = 1.0
LC.register()

import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.scripts.train import TrainConfig  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

STOCK = "Mjlab-Tracking-Flat-Unitree-G1"
DEVICE = "cuda:0"


def build(task, train_dir: Path, n_envs, seed):
  env_yaml = (train_dir / "params" / "env.yaml").read_text()
  motion = next(l.split(":", 1)[1].strip() for l in env_yaml.splitlines() if l.strip().startswith("motion_file:"))
  cfg = TrainConfig.from_task(task)
  configure_torch_backends()
  cfg.env.commands["motion"].motion_file = motion
  cfg.env.scene.num_envs = n_envs
  cfg.agent.seed = seed
  cfg.env.seed = seed
  env = ManagerBasedRlEnv(cfg=cfg.env, device=DEVICE)
  env_w = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  from mjlab.tasks.registry import load_runner_cls
  runner = load_runner_cls(task)(env_w, asdict(cfg.agent), None, DEVICE)
  return env, env_w, runner


def equal(a):
  """Same batch: a stock RewardManager (stock task's reward config, from mjlab's registry) is attached to
  the variant env and computes the stock reward on exactly the state the variant reward is computed on,
  inside every env.step. Phase 1 (lambda = 0): totals must be bitwise equal. Phase 2 (lambda = 0.5, a
  sanity check of the term): total - stock must equal 0.5 * exp(-e_leg^2/sigma^2) * dt."""
  from mjlab.managers.reward_manager import RewardManager
  from mjlab.tasks.registry import load_env_cfg
  stock_cfg, var_cfg = asdict(load_env_cfg(STOCK)), asdict(load_env_cfg(LC.TASK_ID))
  extra = var_cfg["rewards"].pop(LC.TERM)
  import yaml  # compared as YAML dumps, as policy_eval/rollout.py does (scene holds objects without __eq__)
  cfg_same = yaml.dump(stock_cfg, sort_keys=False) == yaml.dump(var_cfg, sort_keys=False)
  env, env_w, runner = build(LC.TASK_ID, a.train_dir, a.envs, 42)
  runner.load(str(a.checkpoint), map_location=DEVICE)
  st = LC.state_of(env)
  vrm = env.reward_manager
  srm = RewardManager(load_env_cfg(STOCK).rewards, env, scale_by_dt=vrm._scale_by_dt)
  assert list(srm.active_terms) == [n for n in vrm.active_terms if n != LC.TERM]
  rec = []
  vcompute = vrm.compute

  def compute(dt):
    stock = srm.compute(dt).clone()
    stock_terms = srm._step_reward.clone()
    total = vcompute(dt)
    li = vrm.active_terms.index(LC.TERM)
    rec.append((st.lam, stock, total.clone(), vrm._step_reward[:, li].clone(),
                (vrm._step_reward[:, [i for i, n in enumerate(vrm.active_terms) if n != LC.TERM]]
                 == stock_terms).float().mean(), dt))
    return total

  vrm.compute = compute
  actor = runner.alg.actor
  actor.eval()
  env_w.episode_length_buf = torch.randint_like(env_w.episode_length_buf, high=int(env_w.max_episode_length))
  obs = env_w.get_observations().to(DEVICE)
  dones_total = 0
  with torch.inference_mode():
    for t in range(2 * a.steps):
      st.lam = 0.0 if t < a.steps else 0.5
      obs, _, dones, _ = env_w.step(actor(obs, stochastic_output=True))
      dones_total += int(dones.sum()) if t < a.steps else 0
  p1, p2 = rec[:a.steps], rec[a.steps:]
  eq = torch.stack([(s == v).float().mean() for _, s, v, _, _, _ in p1]).mean().item()
  dmax = max((s - v).abs().max().item() for _, s, v, _, _, _ in p1)
  leg0 = max(r.abs().max().item() for _, _, _, r, _, _ in p1)
  terms_eq = torch.stack([x for *_, x, _ in p1]).mean().item()
  # phase 2: the leg term adds exactly lambda * exp(.) * dt on top of the stock total
  d2 = max(((v - s) - r * dt).abs().max().item() for _, s, v, r, _, dt in p2)
  r2 = float(torch.stack([r.mean() for _, _, _, r, _, _ in p2]).mean())
  res = {
    "envs": a.envs, "steps_per_phase": a.steps, "checkpoint": str(a.checkpoint), "seed": 42,
    "actions": "stochastic policy, training env (randomization, pushes, obs noise, random initial episode lengths)",
    "variant_cfg_equals_stock_cfg_without_leg_term": cfg_same, "leg_term_cfg": {k: str(v) for k, v in extra.items()},
    "resets_during_phase1": dones_total,
    "phase1_lambda0_total_bitwise_equal_fraction": eq,
    "phase1_lambda0_total_max_abs_diff": dmax,
    "phase1_leg_term_max_abs": leg0,
    "phase1_stock_terms_bitwise_equal_fraction": terms_eq,
    "phase2_lambda0.5_mean_raw_leg_term": r2,
    "phase2_total_minus_stock_minus_leg_term_dt_max_abs": d2,
  }
  res["passed"] = cfg_same and eq == 1.0 and dmax == 0.0 and leg0 == 0.0 and terms_eq == 1.0
  print(json.dumps(res, indent=1))
  out = Path(a.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / "equal_check.json").write_text(json.dumps(res, indent=1))


def eps(a):
  env, env_w, runner = build(LC.TASK_ID, a.train_dir, 4096, 42)
  runner.load(str(a.checkpoint), map_location=DEVICE)
  st = LC.state_of(env)
  assert st.lam == 0.0
  actor = runner.alg.actor
  actor.eval()
  env_w.episode_length_buf = torch.randint_like(env_w.episode_length_buf, high=int(env_w.max_episode_length))
  obs = env_w.get_observations().to(DEVICE)
  rows = []
  with torch.inference_mode():
    for it in range(a.warmup + a.iters):
      st.reset_batch()
      logged = []
      for _ in range(24):  # num_steps_per_env
        obs, _, _, ex = env_w.step(actor(obs, stochastic_output=True))
        logged.append(float(ex["log"]["Metrics/motion/error_body_pos"]))
      m = st.means()
      rows.append({"it": it, "e_leg": m["e_leg"], "e_body": m["e_body"], "e_arms": m["e_arms"],
                   "e_torso": m["e_torso"], "logged_error_body_pos": float(np.mean(logged)),
                   "sampling_entropy": float(ex["log"].get("Metrics/motion/sampling_entropy", np.nan))})
      if it % 10 == 0:
        print({k: round(v, 5) for k, v in rows[-1].items()}, flush=True)
  meas = rows[a.warmup:]
  f = lambda k: float(np.mean([r[k] for r in meas]))
  sd = lambda k: float(np.std([r[k] for r in meas]))
  res = {"checkpoint": str(a.checkpoint), "warmup_iterations": a.warmup, "measured_iterations": a.iters,
         "e_leg_m": f("e_leg"), "e_leg_sd_over_iterations_m": sd("e_leg"), "e_body_m": f("e_body"),
         "logged_error_body_pos_same_rollout": f("logged_error_body_pos"),
         "e_arms_m": f("e_arms"), "e_torso_m": f("e_torso"), "eps_m": 0.9 * f("e_leg"), "rows": rows}
  print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
  Path(a.out).write_text(json.dumps(res, indent=1))


def shares(a):
  """Stock reward at a checkpoint under the training sampling: each stock term's mean per-step value
  (dt-scaled, as summed into the total) and its share of the mean total; plus the motion_body_pos term
  against the body-mean error it actually sees. Runs the variant env at lambda = 0 (= stock, see equal)."""
  env, env_w, runner = build(LC.TASK_ID, a.train_dir, a.envs, 42)
  runner.load(str(a.checkpoint), map_location=DEVICE)
  rm = env.reward_manager
  names = [n for n in rm.active_terms if n != LC.TERM]
  idx = [rm.active_terms.index(n) for n in names]
  bp = rm.active_terms.index("motion_body_pos")
  cmd = env.command_manager.get_term("motion")
  actor = runner.alg.actor
  actor.eval()
  env_w.episode_length_buf = torch.randint_like(env_w.episode_length_buf, high=int(env_w.max_episode_length))
  obs = env_w.get_observations().to(DEVICE)
  sums, tot, n = torch.zeros(len(names), device=DEVICE), 0.0, 0
  bp_val, bp_err, bp_rms = [], [], []
  vcompute = rm.compute

  def compute(dt):
    out = vcompute(dt)
    if record[0]:
      nonlocal tot, n
      sums.add_((rm._step_reward[:, idx] * rm._scale).sum(0))
      tot += float(out.sum())
      n += out.numel()
      d = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1)
      bp_val.append(rm._step_reward[:, bp].cpu())
      bp_err.append(d.mean(-1).cpu())
      bp_rms.append(d.square().mean(-1).sqrt().cpu())
    return out

  record = [False]
  rm.compute = compute
  with torch.inference_mode():
    for it in range(a.warmup + a.iters):
      record[0] = it >= a.warmup
      for _ in range(24):
        obs, _, _, _ = env_w.step(actor(obs, stochastic_output=True))
  mean = (sums / n).cpu().numpy()
  total = tot / n
  bv, be, br = torch.cat(bp_val).numpy(), torch.cat(bp_err).numpy(), torch.cat(bp_rms).numpy()
  dt = env.step_dt
  bins = {}
  for c in (3.0, 5.0):
    m = np.abs(100 * be - c) < 0.25
    bins[f"{c:g}cm"] = {"frames": int(m.sum()), "term_raw_mean": float(bv[m].mean()),
                        "rms_error_cm_mean": float(100 * br[m].mean())}
  std = float(env.cfg.rewards["motion_body_pos"].params["std"])
  w = float(env.cfg.rewards["motion_body_pos"].weight)
  res = {"checkpoint": str(a.checkpoint), "envs": a.envs, "iterations": a.iters, "step_dt": dt,
         "total_per_step": total,
         "terms": {nm: {"weight": float(env.cfg.rewards[nm].weight), "per_step": float(v), "share": float(v / total)}
                   for nm, v in zip(names, mean)},
         "positive_terms_sum_per_step": float(mean[mean > 0].sum()),
         "motion_body_pos": {
           "std_m": std, "weight": w,
           "analytic_uniform_error": {f"{c}cm": {"raw": float(np.exp(-(c / 100) ** 2 / std ** 2)),
                                               "per_step": float(w * dt * np.exp(-(c / 100) ** 2 / std ** 2))}
                                      for c in (3, 5)},
           "measured_at_body_mean_error_within_0.25cm": bins,
           "body_mean_error_cm_mean": float(100 * be.mean()), "term_raw_mean": float(bv.mean())}}
  print(json.dumps(res, indent=1))
  Path(a.out).write_text(json.dumps(res, indent=1))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["equal", "eps", "shares"])
  ap.add_argument("--train_dir", type=Path, required=True)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--envs", type=int, default=1024)
  ap.add_argument("--steps", type=int, default=120)
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--iters", type=int, default=100)
  ap.add_argument("--out")
  ap.add_argument("--out_dir")
  a = ap.parse_args()
  {"equal": equal, "eps": eps, "shares": shares}[a.mode](a)


if __name__ == "__main__":
  main()
