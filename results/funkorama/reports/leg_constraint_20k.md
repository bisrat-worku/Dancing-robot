# G1 tracking policy evaluation: funkorama

Training run `2026-11-19_00-45-34_funkorama`, checkpoint `model_19998.pt`, task `Mjlab-Tracking-Flat-Unitree-G1-LegConstraint`. Generated 2026-11-19T06:32:45 by `policy_eval/evaluate.py`.

## Reference vs executed motion

Executed = the clean run (deterministic policy, no randomization). For the randomized column, each motion-only metric is computed per episode and then averaged over the episodes; for an episode that terminated early, the metric covers the frames before the termination.

| metric | reference | executed (clean) | difference | executed (randomized, mean over episodes) |
|---|---|---|---|---|
| PFC (EDGE formula, G1 bodies, 50 fps) | 0.071 | 0.006 | -0.064 | 0.007 |
| Beat alignment, kinematic->music (primary) | 0.141 | 0.180 | 0.039 | 0.170 |
| Beat alignment, music->kinematic (Bailando/EDGE) | 0.163 | 0.212 | 0.050 | 0.197 |
|   chance level of that score (random phase of the beat grid) | 0.192 | 0.184 | -0.008 | n/a |
| number of kinematic beats | 51 | 52 | 1 | n/a |
| joint jerk, mean [rad/s^3] | 920 | 489 | -431 | 663 |
| joint jerk, p95 [rad/s^3] | 3362 | 1689 | -1674 | n/a |
| frames with any joint near a position limit | 0.352 | 0.043 | -0.309 | 0.032 |
| joint-frames near a position limit (mean over joints) | 0.014 | 0.002 | -0.013 | n/a |
| frames with any joint beyond its position range | 0.000 | 0.000 | 0.000 | n/a |
| frames with any joint near a velocity limit | 0.000 | 0.000 | 0.000 | 0.000 |
| frames with any joint beyond its velocity limit | 0.000 | 0.000 | 0.000 | n/a |
| peak joint speed [rad/s] | 25.3 | 17.9 | -7.4 | n/a |
| lag of executed behind reference [ms] | 0 (by definition) | 0 (sub-frame -5.2, r = 0.885) | | 0.0 |
| left foot slip while in contact, mean [cm/s] | heel 4.8 / toe 8.2 (EDGE contact labels, 30 fps) | ankle 7.6 / sole-min 4.9 (simulator contact) | not comparable | |
| right foot slip while in contact, mean [cm/s] | heel 6.2 / toe 8.2 (EDGE contact labels, 30 fps) | ankle 7.6 / sole-min 5.1 (simulator contact) | not comparable | |

Foot slip is not like-for-like: the reference has no simulator, so its number comes from `g1/eval_g1_motion.py` on `_g1_fixed.pkl` (30 fps, before the 12 Hz filter) with EDGE's predicted contacts. The executed number uses MuJoCo's foot/floor contacts.

### Wrong-song calibration of the beat alignment score (kinematic->music)

| music | reference | executed (clean) |
|---|---|---|
| **funkorama (true alignment, 7.5 s)** | 0.1408 | 0.1802 |
| librosa:vibeace | 0.3590 | 0.2683 |
| librosa:fishin | 0.2717 | 0.2140 |
| librosa:nutcracker | 0.1516 | 0.1939 |
| self:offset=0.0 | 0.2049 | 0.1427 |

## Completion

| protocol | episodes | success rate | completion fraction (mean / min) | first-termination conditions |
|---|---|---|---|---|
| clean (no randomization) | 1 | 1.00 (1/1) | 1.000 / n/a | none |
| randomized (training reset perturbation + startup DR + obs noise) | 50 | 1.00 (50/50) | 1.000 / 1.000 | none |
| open-loop reference joint targets (no policy, clean) | 1 | 0.00 (0/1) | 0.019 / n/a | ee_body_pos: 1 |
| randomized + training pushes | 50 | 1.00 (50/50) | 1.000 / 1.000 | none |
| clean, foot friction = 0.3 | 1 | 1.00 (1/1) | 1.000 / n/a | none |
| clean, foot friction = 0.6 | 1 | 1.00 (1/1) | 1.000 / n/a | none |
| clean, foot friction = 1.2 | 1 | 1.00 (1/1) | 1.000 / n/a | none |

Open-loop baseline: the robot survives **0.56 s** of 29.96 s (terminated by ee_body_pos).

Fraction of randomized episodes that complete the clip with the body-mean local position error never above the threshold: 5 cm: 0.00, 10 cm: 0.96, 15 cm: 1.00.

## Tracking error

Statistics over every scored frame (frames 1 to the end, or to the first termination). *Averaged* = the mean over the 14 tracked bodies (29 joints for joint error) at each frame, then the statistic over frames. Randomized = all frames of all 50 episodes pooled.

| error | run | mean | median | p95 | max |
|---|---|---|---|---|---|
| position, local [cm] | clean | 2.58 | 2.30 | 5.45 | 7.84 |
| position, local [cm] | randomized | 2.82 | 2.51 | 5.72 | 10.05 |
| position, global [cm] | clean | 5.68 | 5.20 | 11.47 | 13.65 |
| position, global [cm] | randomized | 6.55 | 5.52 | 13.60 | 66.13 |
| orientation, local [deg] | clean | 6.13 | 5.76 | 10.39 | 15.33 |
| orientation, local [deg] | randomized | 6.57 | 6.18 | 11.20 | 24.78 |
| orientation, global [deg] | clean | 6.58 | 6.28 | 10.81 | 16.61 |
| orientation, global [deg] | randomized | 7.13 | 6.70 | 11.97 | 31.16 |
| linear velocity, global [m/s] | clean | 0.23 | 0.20 | 0.50 | 1.38 |
| linear velocity, global [m/s] | randomized | 0.25 | 0.21 | 0.54 | 1.46 |
| linear velocity, local [m/s] | clean | 0.23 | 0.20 | 0.50 | 1.39 |
| linear velocity, local [m/s] | randomized | 0.25 | 0.21 | 0.54 | 1.47 |
| acceleration, global [m/s^2] (2nd priority) | clean | 4.85 | 4.20 | 10.37 | 48.67 |
| acceleration, global [m/s^2] (2nd priority) | randomized | 5.12 | 4.41 | 10.86 | 49.73 |
| joint position, mean over joints [deg] | clean | 4.01 | 3.56 | 6.91 | 11.11 |
| joint position, mean over joints [deg] | randomized | 4.15 | 3.73 | 7.17 | 11.69 |

Time-warped (DTW) global position error, clean run: 5.59 cm (band ±25 frames) vs 5.68 cm without warping.


### Per body (local position error, cm)

| body | clean mean | clean p95 | clean max | randomized mean | randomized p95 | randomized max |
|---|---|---|---|---|---|---|
| pelvis | 0.86 | 2.37 | 5.68 | 0.91 | 2.51 | 7.13 |
| left_hip_roll_link | 1.51 | 3.70 | 6.24 | 1.59 | 3.77 | 7.15 |
| left_knee_link | 3.22 | 7.98 | 12.50 | 3.44 | 8.04 | 17.57 |
| left_ankle_roll_link | 4.81 | 10.55 | 19.04 | 5.38 | 11.55 | 30.95 |
| right_hip_roll_link | 1.87 | 4.98 | 9.24 | 1.93 | 5.28 | 10.17 |
| right_knee_link | 3.63 | 10.20 | 18.85 | 3.99 | 11.19 | 20.85 |
| right_ankle_roll_link | 5.53 | 15.12 | 31.61 | 6.26 | 16.52 | 34.47 |
| torso_link | 0.72 | 2.21 | 5.61 | 0.77 | 2.35 | 7.05 |
| left_shoulder_roll_link | 1.20 | 2.69 | 5.95 | 1.46 | 3.06 | 7.53 |
| left_elbow_link | 2.25 | 4.94 | 11.37 | 2.41 | 5.15 | 12.56 |
| left_wrist_yaw_link | 3.39 | 7.52 | 14.56 | 3.55 | 7.84 | 19.01 |
| right_shoulder_roll_link | 1.25 | 2.98 | 6.78 | 1.51 | 3.35 | 9.10 |
| right_elbow_link | 2.40 | 5.30 | 7.75 | 2.58 | 5.50 | 15.74 |
| right_wrist_yaw_link | 3.50 | 7.85 | 11.59 | 3.72 | 8.02 | 17.87 |

Full per-body tables (global position, orientation, velocity, per-joint error) are in `metrics.json`.

## Per-body error against training iteration

Training logged no per-body error (WandB and TensorBoard hold only body averages), so the clean evaluation was rerun on every saved checkpoint. See `plots/per_body_error_vs_iteration.png`.

| iteration | completed | completion | body-mean local pos err [cm] | legs | arms | torso |
|---|---|---|---|---|---|---|
| 10000 | yes | 1.000 | 3.17 | 4.42 | 2.64 | 1.01 |
| 10500 | yes | 1.000 | 2.74 | 3.35 | 2.76 | 0.84 |
| 11000 | yes | 1.000 | 2.70 | 3.48 | 2.54 | 0.85 |
| 11500 | yes | 1.000 | 2.74 | 3.46 | 2.64 | 0.87 |
| 12000 | yes | 1.000 | 2.74 | 3.49 | 2.62 | 0.84 |
| 12500 | yes | 1.000 | 2.70 | 3.51 | 2.50 | 0.85 |
| 13000 | yes | 1.000 | 2.73 | 3.50 | 2.59 | 0.84 |
| 13500 | yes | 1.000 | 2.69 | 3.44 | 2.56 | 0.87 |
| 14000 | yes | 1.000 | 2.69 | 3.46 | 2.52 | 0.86 |
| 14500 | yes | 1.000 | 2.69 | 3.48 | 2.52 | 0.83 |
| 15000 | yes | 1.000 | 2.65 | 3.45 | 2.46 | 0.81 |
| 15500 | yes | 1.000 | 2.72 | 3.55 | 2.53 | 0.83 |
| 16000 | yes | 1.000 | 2.67 | 3.46 | 2.50 | 0.84 |
| 16500 | yes | 1.000 | 2.64 | 3.52 | 2.39 | 0.80 |
| 17000 | yes | 1.000 | 2.68 | 3.53 | 2.46 | 0.79 |
| 17500 | yes | 1.000 | 2.62 | 3.38 | 2.45 | 0.84 |
| 18000 | yes | 1.000 | 2.64 | 3.44 | 2.45 | 0.79 |
| 18500 | yes | 1.000 | 2.62 | 3.39 | 2.45 | 0.82 |
| 19000 | yes | 1.000 | 2.58 | 3.29 | 2.47 | 0.78 |
| 19500 | yes | 1.000 | 2.68 | 3.52 | 2.45 | 0.83 |
| 19998 | yes | 1.000 | 2.57 | 3.41 | 2.33 | 0.78 |

## Effort and smoothness (executed)

| metric | clean | randomized (mean over episodes) |
|---|---|---|
| action rate (sum over joints of squared action change per step), mean | 0.528 | 0.816 |
| mean abs torque [Nm] | 3.92 | 4.09 |
| peak abs torque [Nm] | 68.69 | 70.86 |
| highest peak torque / limit over joints | 0.955 | |
| substeps with any joint at its torque limit | 0.0000 | 0.0016 |
| joint-substeps at the torque limit (mean over joints) | 0.0000 | |

Joints most often at the torque limit (clean): left_hip_pitch_joint 0.0000, left_hip_roll_joint 0.0000, left_hip_yaw_joint 0.0000, left_knee_joint 0.0000, left_ankle_pitch_joint 0.0000.

## Training cost

- Wall clock: 4.83 h (2026-11-19T00:45:43 to 2026-11-19T05:35:27), 10000 iterations, 4096 environments, 24 steps per env per iteration.
- Environment steps: 983,040,000 total; 57938 steps/s (mean of the logged Perf/total_fps), 56548 steps/s overall including logging and checkpointing.
- First checkpoint that completes the clean run: **10000** (≈ 4.83 h into training).
- Training seeds: 1 finished run(s) with this budget (seed 42). Only one training seed exists, so there is no seed spread to report. Other runs in the folder, not counted: `2026-11-12_20-39-12_funkorama` (20 iterations planned, 512 envs, last checkpoint 19), `2026-11-12_20-39-59_funkorama` (10000 iterations planned, 4096 envs, last checkpoint 0), `2026-11-12_20-43-24_funkorama` (10000 iterations planned, 4096 envs, last checkpoint 9999), `2026-11-17_20-06-34_funkorama` (10000 iterations planned, 4096 envs, last checkpoint 19998), `2026-11-18_23-59-29_funkorama` (10000 iterations planned, 4096 envs, last checkpoint 11500).

## Metric self-tests

| test | result | detail |
|---|---|---|
| reference vs itself: zero error | PASS | max over frames/bodies: pos_local_cm=0.00e+00, pos_global_cm=0.00e+00, ori_local_deg=3.18e-15, ori_global_deg=3.18e-15, linvel_global=0.00e+00, linvel_local=0.00e+00, acc_global=0.00e+00, joint_abs_deg=0.00e+00, joint_l2_rad=0.00e+00 |
| reference vs itself: zero lag | PASS | lag=0 frames (sub-frame 0), corr=1.000000 |
| reference shifted by +7 frames: lag = +7 | PASS | lag=7 frames = 140 ms (sub-frame 7.01), corr=1.0000 |
| reference shifted by -4 frames: lag = -4 | PASS | lag=-4 frames = -80 ms (sub-frame -4.00), corr=1.0000 |
| reference shifted by +1 frames: lag = +1 | PASS | lag=1 frames = 20 ms (sub-frame 1.01), corr=1.0000 |
| wrong song (librosa:vibeace) scores lower than the right song (reference motion) | **FAIL** | kin->music right 0.1408 vs wrong 0.3590; music->kin right 0.1627 vs wrong 0.2807 |
| wrong song (librosa:fishin) scores lower than the right song (reference motion) | **FAIL** | kin->music right 0.1408 vs wrong 0.2717; music->kin right 0.1627 vs wrong 0.2430 |
| wrong song (librosa:nutcracker) scores lower than the right song (reference motion) | **FAIL** | kin->music right 0.1408 vs wrong 0.1516; music->kin right 0.1627 vs wrong 0.1643 |
| wrong song (self:offset=0.0) scores lower than the right song (reference motion) | **FAIL** | kin->music right 0.1408 vs wrong 0.2049; music->kin right 0.1627 vs wrong 0.2173 |
| positive control: synthetic motion with speed minima on the song's beats scores above its chance mean on the right song and lower on every wrong song | PASS | right 0.8587 (chance mean 0.1926; 49 kinematic beats for 44 music beats, the extra ones lie before the first / after the last music beat); wrong: librosa:vibeace 0.2589, librosa:fishin 0.2910, librosa:nutcracker 0.2201, self:offset=0.0 0.0001. For the reference motion itself: right-song score 0.1408 vs chance 0.1925 (p5 0.1259, p95 0.2580) |
| left arm offset by 10 cm shows up only in the left-arm bodies | PASS | pelvis: local 0.000 / global 0.000 cm; left_hip_roll_link: local 0.000 / global 0.000 cm; left_knee_link: local 0.000 / global 0.000 cm; left_ankle_roll_link: local 0.000 / global 0.000 cm; right_hip_roll_link: local 0.000 / global 0.000 cm; right_knee_link: local 0.000 / global 0.000 cm; right_ankle_roll_link: local 0.000 / global 0.000 cm; torso_link: local 0.000 / global 0.000 cm; left_shoulder_roll_link: local 10.000 / global 10.000 cm; left_elbow_link: local 10.000 / global 10.000 cm; left_wrist_yaw_link: local 10.000 / global 10.000 cm; right_shoulder_roll_link: local 0.000 / global 0.000 cm; right_elbow_link: local 0.000 / global 0.000 cm; right_wrist_yaw_link: local 0.000 / global 0.000 cm |
| numpy anchor-relative reference equals mjlab's body_pos_relative_w / body_quat_relative_w | PASS | max |dpos| = 0.0001 mm, max angle = 0.0000 deg over 1498 frames |

## Run-to-run variation of the clean run

MuJoCo Warp on the GPU is not bit-deterministic (mujoco_warp issue 562, noted in mjlab's `seed_rng`). Two identical clean rollouts (the second with video rendering) agree exactly at first and then drift apart: max difference 16 mm in body position and 5.41 deg in joint position over the clip. The clean run was therefore repeated 5 times. The tables above show repeat 0. The spread over all repeats is:

| metric | min | mean | max |
|---|---|---|---|
| completion_fraction | 1.0000 | 1.0000 | 1.0000 |
| pos_local_cm_mean | 2.5736 | 2.5811 | 2.5881 |
| pos_global_cm_mean | 5.5528 | 5.6326 | 5.7181 |
| ori_local_deg_mean | 6.1210 | 6.1301 | 6.1404 |
| ori_global_deg_mean | 6.5561 | 6.5809 | 6.6109 |
| linvel_global_mean | 0.2295 | 0.2297 | 0.2300 |
| joint_abs_deg_mean | 4.0092 | 4.0126 | 4.0165 |
| lag_ms | 0.0000 | 0.0000 | 0.0000 |
| torque_mean_abs | 3.9132 | 3.9219 | 3.9280 |
| torque_sat_frac_any_joint | 0.0000 | 0.0000 | 0.0000 |
| pfc_exec | 0.0064 | 0.0066 | 0.0067 |
| bas_exec_kin_to_music | 0.1785 | 0.1828 | 0.1919 |
| slip_left_sole_min_mean | 4.9420 | 5.0627 | 5.1695 |
| slip_right_sole_min_mean | 5.0639 | 5.2566 | 5.5634 |

Completed the clip in 5/5 repeats. The same applies to the randomized episodes: a seed fixes the perturbation, DR draw and observation noise, but rerunning that seed does not reproduce the trajectory bit for bit.

## Metric definitions

All metrics run on the 50 Hz policy grid. Frame *k* of an episode is the robot state after env step *k*,
compared with reference frame *k* (checked at run time: mjlab scores step *k* against command frame *k*).
Frame 0 is the reset state, which `MotionCommand` writes from reference frame 0, with joint positions
clipped to the soft joint limits (90 % of the range). It is excluded from all error statistics.
Unless stated otherwise, tracking errors cover the 14 bodies the task tracks: `pelvis`, `left_hip_roll_link`, `left_knee_link`, `left_ankle_roll_link`, `right_hip_roll_link`, `right_knee_link`, `right_ankle_roll_link`, `torso_link`, `left_shoulder_roll_link`, `left_elbow_link`, `left_wrist_yaw_link`, `right_shoulder_roll_link`, `right_elbow_link`, `right_wrist_yaw_link`.

- **Completion.** An episode succeeds when it reaches the last clip frame (1498, 29.96 s)
  without any training termination term firing (`anchor_pos`, `anchor_ori`, `ee_body_pos`; `time_out` is
  disabled by setting the episode length to 1e9 s). The *first-termination time* is k/50 s for the step k
  where a term first fires. The *completion fraction* is (last scored frame)/(last clip frame). The
  terminal frame is kept and scored. Nothing after it is recorded, because `auto_reset` is off.
- **Local error (training reward frame).** This is mjlab `MotionCommand.update_relative_body_poses`. The
  reference is translated so its anchor (`torso_link`) sits at the robot anchor's x, y, with the
  reference anchor's own z, then turned by the yaw of q_robot_anchor · q_ref_anchor⁻¹. Position error is
  ‖p_robot − p_ref,rel‖ (cm). Orientation error is the rotation angle between q_robot and q_ref,rel (deg).
  Self-test 5 checks this numpy implementation against the values mjlab computed inside the env.
- **Global error.** The same quantities in the world frame, with no re-anchoring (root drift included).
- **Linear velocity error.** Global: ‖v_robot − v_ref‖ (m/s), the training term `motion_body_lin_vel`.
  Local: the reference velocity is first turned by the same heading difference.
- **Joint position error.** |q_robot − q_ref| per joint (deg). The L2 norm over joints in rad, the
  training log's `error_joint_pos`, is in `metrics.json`.
- **Acceleration error (second priority).** ‖a_robot − a_ref‖, with a = central difference of body linear
  velocity (m/s²).
- **DTW error (second priority).** Dynamic time warping between the reference and executed sequences of
  the 14 global body positions. The frame cost is the mean body distance (cm), the Sakoe-Chiba band is ±25
  frames (0.5 s), and the steps are (1,0), (0,1), (1,1). The result is the mean cost along the optimal path.
- **Stays under X cm.** The episode completes and the per-frame body-mean local position error is ≤ X on
  every frame.
- **PFC.** EDGE `eval/eval_pfc.py`, unchanged except for the body mapping. SMPL pelvis → G1 `pelvis`;
  SMPL ankles (7, 8) → `left/right_ankle_roll_link` origin; SMPL feet/toes (10, 11) → the front tip of the
  G1 sole, i.e. the front-most end point of the `*_foot*_collision` capsules, at [0.132, 0.0, -0.025] m in the
  `ankle_roll_link` frame. DT = 1/50. Up axis z. Root acceleration normalised by its maximum over the clip.
  Foot speeds are per frame, as in EDGE, so PFC depends on the frame rate: the same motion scores
  (30/50)² = 0.36 × its 30 fps value. These numbers are therefore not comparable with EDGE's paper
  values, which are at 30 fps. For context, the source EDGE motion (SMPL joints, 30 fps, EDGE's own code
  unchanged) scores 0.5524.
- **Foot slip (executed).** For each foot, the frames where MuJoCo reports contact between that foot's
  subtree and the terrain (a passive contact sensor added for the evaluation). Two speeds are reported:
  the horizontal speed of the `ankle_roll_link` origin (the literal spec), and the smaller of the
  horizontal speeds of the sole's heel and toe tips (rigid-body velocity v + ω × r). The second does not
  count a foot pivoting on its heel or toe as slip. Also reported: the fraction of contact frames above
  10.0 cm/s.
- **Foot slip (reference).** `g1/eval_g1_motion.py` (the existing checks script) run unchanged on
  `<song>_g1_fixed.pkl`: G1 heel/toe speed while EDGE's predicted contact is > 0.95.
- **Beat alignment.**
  - Music beats: librosa 1.0.0 `beat_track` on the onset envelope of the song from 7.5 s for the
    clip's duration, at sr = 50 × 512 so that one librosa frame is one motion frame (MINT
    `tools/preprocessing.py`, with tightness 100 and start_bpm = librosa's own tempo estimate).
  - Kinematic beats: local minima (`argrelextrema`, `np.less`) of the kinetic velocity, which is the mean
    over all 30 G1 bodies of the per-frame displacement norm, Gaussian-smoothed with σ = 0.083 s
    (4.17 frames). This is Bailando `utils/beat_align_score.py`'s σ = 5 frames at 60 fps.
  - Score (primary, as specified and as in the AIST++ paper): the mean over kinematic beats of
    exp(−d²/2σ²), where d is the distance in frames to the nearest music beat and σ = 0.05 s = 2.5
    frames.
  - Also reported: Bailando's `BA()` direction, the mean over music beats of the same kernel to the
    nearest kinematic beat. EDGE's paper numbers use this direction.
  - The file `tools/calculate_beat_scores.py` doesn't exist in google-research/mint (checked against the
    repo tree and its history), so the port follows the paper definition, Bailando's implementation and
    MINT's audio preprocessing.
- **Lag.** The integer shift in [−25, 25] frames (±0.5 s) that maximises the Pearson correlation between
  the reference and executed unsmoothed kinetic velocity curves over their overlap. Positive means the
  executed motion is late. The sub-frame value comes from a parabolic fit through the peak.
- **Joint jerk.** |Δ³q|·fps³ per joint and frame (rad/s³), mean and 95th percentile over joints and frames.
- **Action rate.** Σ_j (a_k − a_{k−1})² per step over the raw policy actions, for steps 2 onward. Step 1
  is excluded because mjlab's last action is zero at reset. This matches the training term `action_rate_l2`.
- **Torque.** MuJoCo `actuator_force`, recorded at every physics substep (4 per policy step, 200 Hz).
  *At the limit* means |τ| ≥ 0.99 × the actuator `forcerange`.
- **Near a joint limit.** Within 3° of the MuJoCo joint range (hard limit).
  **Near a velocity limit** means |q̇| ≥ 0.9 × the motor velocity limit from mjlab's `g1_constants.py`.
  mjlab doesn't enforce these limits in simulation.


## Thresholds and where they come from

| parameter | value | source |
|---|---|---|
| anchor_pos termination | |z_torso - z_torso,ref| > 0.25 m | training config (params/env.yaml) |
| anchor_ori termination | |g_z,ref - g_z| > 0.8 (projected gravity, torso frame) | training config |
| ee_body_pos termination | |z - z_ref,rel| > 0.25 m on either ankle_roll or wrist_yaw link | training config |
| position-error thresholds | 5 cm, 10 cm, 15 cm | chosen for this evaluation (no external standard); body-mean local error, every frame |
| joint position limit margin | 3 deg | same as g1/eval_g1_motion.py |
| velocity limit margin | 0.1 x limit | chosen for this evaluation |
| velocity limits | 5020: 37, 7520-14: 32, 7520-22: 20, 4010: 22 rad/s | mjlab g1_constants.py |
| torque limit | |tau| >= 0.99 x forcerange | forcerange from the MuJoCo model (mjlab effort_limit) |
| foot slip speed | 10 cm/s | same as g1/eval_g1_motion.py |
| beat sigma | 0.05 s | AIST++ / Bailando: 3 frames at 60 fps |
| kinetic velocity smoothing | 0.0833 s | Bailando: gaussian_filter sigma 5 at 60 fps |
| music analysis | hop 512, sr = fps x hop, tightness 100 | MINT tools/preprocessing.py |
| lag search window | ±0.5 s | chosen for this evaluation |
| randomized episodes / seeds | 50, seeds from 1000 | specification |
| reset perturbation (randomized) | pose ±5 cm xy, ±1 cm z, ±0.1 rad roll/pitch, ±0.2 rad yaw; velocity as push range; joints ±0.1 rad | training config |
| startup DR (randomized) | torso CoM ±2.5/5/5 cm, encoder bias ±0.01 rad, foot friction 0.3-1.2 | training config |
| observation noise (randomized) | training actor noise (anchor pos ±0.25, ori ±0.05, lin vel ±0.5, ang vel ±0.2, q ±0.01, qd ±0.5) | training config |
| pushes (second priority) | every 1-3 s, velocity kick ±0.5 m/s xy, ±0.2 z, ±0.52 rad/s roll/pitch, ±0.78 yaw | training config |
| friction sweep | 0.3, 0.6, 1.2 | training DR range ends and the model default |

## Provenance

- **Motion version**: foot-fixed G1 retarget `funkorama_g1_fixed.pkl` (g1/fix_g1_feet.py) → g1/edge_to_mjlab.py: 12 Hz low-pass, resampled 30 → 50 fps, frames clipped to joint limits. From the training log: == [3/4] building mjlab reference (smooth 12 Hz) / [in]  outputs/funkorama_20261107_215248/funkorama_g1_fixed.pkl  (900 frames @ 30 fps = 30.0 s) / Motion interpolated, input frames: 900, input fps: 30.0, output frames: 1499, output fps: 50.0 / smoothing: 12 Hz;  peak joint speed 32.2 -> 28.0 rad/s;  frames clipped to joint limits: 76 / 
- Reference file `outputs/funkorama_20261107_215248/funkorama_mjlab_motion.npz`, sha256 `54c6df5a3e12ba73…`; source `outputs/funkorama_20261107_215248/funkorama_g1_fixed.pkl` sha256 `29bb02ed6a1f48ca…`.
- Checkpoint `outputs/funkorama_20261107_215248/rl/g1_tracking/2026-11-19_00-45-34_funkorama/model_19998.pt`, sha256 `5d7645068a816d77…`.
- Audio `custom_music/funkorama.wav` (sha256 `c521221dbc6645f9…`): motion frame 0 = 7.5 s into the song. Measured by cross-correlating the audio track of EDGE's render (test_funkorama.mp4) with the wav: best match at 7.5 s, normalised correlation 1.000. EDGE's test.py picks a random window of 2.5 s slices; with seed.txt = 0 it starts at slice 3.
- Randomized seeds: 1000..1049 (seed i -> torch, numpy, python and warp RNGs reseeded before the episode's startup DR and reset). Push seeds: 11000..11049.
- Training config check: the env config rebuilt from mjlab's registry is identical to the run's params/env.yaml (YAML dump compared as text)
- Edge repo commit `404729cb542c32c847d5431a6f0cf51627b56556` (uncommitted changes: policy_eval/config.py, policy_eval/evaluate.py, policy_eval/rollout.py, policy_eval/compare_runs.py); mjlab commit `d0ad1fe1c9b780c1b4d06f3fbf4407ab51e92313` (clean); training-time mjlab commit `None`.
- Library versions: mjlab 1.6.0, mujoco 3.16.0, mujoco-warp 3.16.0, warp-lang 1.18.0, torch 2.14.0, rsl-rl-lib 5.5.2, numpy 2.5.1, scipy 1.16.2, matplotlib 3.10.8, tensorboard 2.20.0, python 3.13.18; GPU NVIDIA GeForce RTX 5060 Laptop GPU.

## Not computed, or computed differently than specified

- Only one complete training seed exists (seed 42), so no seed spread can be reported.
- Training scored the stochastic policy on 10 s windows with adaptive start frames and pushes; this evaluation scores the deterministic policy on the full clip from frame 0, so its errors are not the same quantity as the training log's Metrics/motion/* values (those are listed in metrics.json under training.final_logged).
- Foot slip on the reference cannot use simulator contacts (there is no simulation). The reference number is the existing checks script on the 30 fps foot-fixed retarget with EDGE contact labels, so reference and executed slip are reported side by side but are different measurements.
- Action rate and torque exist only for the executed motion (the reference has no actions; inverse dynamics of the reference was not requested and was not computed).
- PFC is computed at 50 fps on G1 bodies; it is not comparable with EDGE's published (30 fps, SMPL) values. The source EDGE motion's own 30 fps SMPL PFC is given for context only.
- The specified MINT script tools/calculate_beat_scores.py does not exist in google-research/mint; the beat alignment port follows the AIST++ paper definition (kinematic->music) and reports Bailando's direction too.
- MINT takes start_bpm from the AIST++ file name; here it is librosa's own tempo estimate of the clip.
- Velocity limits are the motor limits in mjlab's g1_constants.py; for the ankle and waist-pitch/roll linkages mjlab assumes a 1:1 ratio, and the 5020's 37 rad/s is used for them.
