# stock 20k vs leg constraint 20k

- a = `outputs/funkorama_20261107_215248/rl/g1_tracking/2026-11-17_20-06-34_funkorama/model_19998.pt` (task `Mjlab-Tracking-Flat-Unitree-G1`)
- b = `outputs/funkorama_20261107_215248/rl/g1_tracking/2026-11-19_00-45-34_funkorama/model_19998.pt` (task `Mjlab-Tracking-Flat-Unitree-G1-LegConstraint`)
- code: commit `08d3172 (eval), e0828fc/7d3f97e (variant)`
- multiplier: {"iterations": [9999, 19998], "lambda_final": 0.098031, "lambda_max_seen": 0.531723, "iterations_at_cap": 0, "first_iteration_at_cap": null, "e_bar_final_cm": 4.2046, "e_batch_last100_cm": 4.234359300000001, "eps_cm": 4.247, "e_body_last100_cm": 3.2133516, "stock_error_body_pos_last100_cm": 3.458925753831863}

### Clean run (no randomization, 1 rollout; the 5 repeats' body-mean range is given)

| metric | stock 20k | leg constraint 20k | b − a | paired test (b vs a) |
|---|---|---|---|---|
| legs local error [cm] | 4.01 | 3.43 | -0.58 |  |
| arms local error [cm] | 2.30 | 2.33 | +0.03 |  |
| torso local error [cm] | 0.91 | 0.79 | -0.12 |  |
| body-mean local error [cm] | 2.84 | 2.58 | -0.25 |  |
| body-mean global error [cm] | 5.04 | 5.68 | +0.64 |  |
| completion (mean fraction) | 1.000 | 1.000 | +0.000 |  |
| joint jerk [rad/s³] | 473 | 489 | +16 |  |
| action rate (L2) | 0.505 | 0.528 | +0.023 |  |
| torque saturation (frac. frames, any joint) | 0.0012 | 0.0000 | -0.0012 |  |
| body-mean local, 5 repeats (min–max) | 2.81–2.84 | 2.57–2.59 | | |

### Randomized set (50 seeds: startup DR, RSI, obs noise; no pushes)

| metric | stock 20k | leg constraint 20k | b − a | paired test (b vs a) |
|---|---|---|---|---|
| legs local error [cm] | 4.33 | 3.77 | -0.57 | median Δ -0.57, b lower in 100%, p = 1.8e-15 |
| arms local error [cm] | 2.52 | 2.54 | +0.02 | median Δ +0.02, b lower in 28%, p = 0.00062 |
| torso local error [cm] | 0.95 | 0.84 | -0.11 | median Δ -0.11, b lower in 100%, p = 1.8e-15 |
| body-mean local error [cm] | 3.07 | 2.82 | -0.25 | median Δ -0.25, b lower in 100%, p = 1.8e-15 |
| body-mean global error [cm] | 6.77 | 6.55 | -0.22 | median Δ -0.18, b lower in 64%, p = 0.011 |
| completion (mean fraction) | 1.000 | 1.000 | +0.000 | median Δ +0.000, b lower in 0%, p = 1 |
| episodes completing the clip | 1.00 | 1.00 | +0.00 |  |
| joint jerk [rad/s³] | 655 | 663 | +8 | median Δ +5.83, b lower in 20%, p = 3.1e-08 |
| action rate (L2) | 0.787 | 0.816 | +0.029 | median Δ +0.029, b lower in 2%, p = 3.6e-15 |
| torque saturation (frac. frames, any joint) | 0.0020 | 0.0016 | -0.0004 | median Δ -0.0003, b lower in 64%, p = 0.02 |

### Push set (50 seeds: randomized + pushes)

| metric | stock 20k | leg constraint 20k | b − a | paired test (b vs a) |
|---|---|---|---|---|
| legs local error [cm] | 4.73 | 4.20 | -0.53 | median Δ -0.51, b lower in 100%, p = 1.8e-15 |
| arms local error [cm] | 2.65 | 2.68 | +0.03 | median Δ +0.02, b lower in 28%, p = 0.0003 |
| torso local error [cm] | 1.02 | 0.92 | -0.10 | median Δ -0.10, b lower in 100%, p = 1.8e-15 |
| body-mean local error [cm] | 3.31 | 3.08 | -0.23 | median Δ -0.23, b lower in 100%, p = 1.8e-15 |
| body-mean global error [cm] | 11.06 | 11.15 | +0.09 | median Δ +0.09, b lower in 42%, p = 0.33 |
| completion (mean fraction) | 1.000 | 1.000 | +0.000 | median Δ +0.000, b lower in 0%, p = 1 |
| episodes completing the clip | 1.00 | 1.00 | +0.00 |  |
| joint jerk [rad/s³] | 709 | 719 | +11 | median Δ +11.93, b lower in 24%, p = 5.2e-07 |
| action rate (L2) | 0.870 | 0.903 | +0.033 | median Δ +0.038, b lower in 10%, p = 2.6e-12 |
| torque saturation (frac. frames, any joint) | 0.0037 | 0.0030 | -0.0007 | median Δ -0.0008, b lower in 64%, p = 0.024 |
