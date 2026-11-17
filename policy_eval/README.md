# policy_eval: evaluation of the mjlab G1 tracking policy

```bash
python3 policy_eval/evaluate.py outputs/<run>/                # full protocol, about 1 h on the RTX 5060
python3 policy_eval/evaluate.py outputs/<run>/ --no-second    # without pushes, friction sweep and DTW
python3 policy_eval/evaluate.py outputs/<run>/ --rescore      # recompute metrics from the saved rollouts
python3 policy_eval/evaluate.py outputs/<run>/ --train_dir <run>/rl/g1_tracking/<dir>   # another training seed
```

Output goes to `outputs/<run>/eval/<training-run>/`:

- `report.md`: the reference-vs-executed table, every definition and threshold, provenance and caveats.
- `metrics.json`: everything in the report plus per-body, per-joint and per-episode detail.
- `episodes.csv`: one row per episode.
- `plots/`: error over time, per-body error, per-body error against iteration, first-termination
  times, kinetic velocity against the beats.
- `clean_run.mp4`: the clean run with the song's audio. The ghost is the reference.
- `raw/*.npz`: the recorded trajectories, so the metrics can be recomputed without simulating.

## Files

| file | venv | role |
|---|---|---|
| `config.py` | any | every threshold, margin, seed and window, each with where it comes from |
| `rollout.py` | mjlab | builds the env from mjlab's registry, checks it against `params/env.yaml`, rolls out and records trajectories |
| `metrics.py` | any | pure numpy: tracking errors in the reward frame and the world frame, PFC, beats, lag, jerk, limits |
| `music_beats.py` | EDGE | librosa music beats on the motion's frame grid |
| `evaluate.py` | mjlab | runs everything above, the self-tests, plots and the report |
| `plot_error_vs_speed.py` | mjlab | per-body error over time against the reference's limb joint speed, plus scatters with correlation and lag |
| `error_diagnostics.py` | mjlab | where the error comes from: speed (fastest 10 % vs rest), start time (needs `rollout.py --start_frame` runs), foot contact and slip, clipped frames and speed spikes |
| `edge_bas.py` | EDGE | beat alignment of raw EDGE motions (from `generate_dance.sh` runs) and of real AIST++ dances (`--aist_gt`), with chance and wrong-song controls |

## Rules this module keeps

- It never changes the training config, rewards, reference motion or checkpoints. Evaluation switches
  are applied to an in-memory copy of the training env config, and before anything runs that config is
  checked to be byte-identical to the run's saved `params/env.yaml`.
- Every metric that can be computed on a motion alone runs through the same function for the reference
  and for the executed motion.
- Self-tests run on every evaluation, and the report lists their results.
