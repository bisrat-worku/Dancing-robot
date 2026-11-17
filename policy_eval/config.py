"""Every number the evaluation depends on, with where it comes from.

Change a value here (or override it with --set key=value on evaluate.py) and it is written into
metrics.json and the report, so two runs are only compared when their configs match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EvalConfig:
  # ---- protocol -------------------------------------------------------------------------------
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  """mjlab task id the policy was trained on (stock config, loaded from mjlab's registry and
  checked field by field against the run's params/env.yaml before anything runs)."""
  clean_repeats: int = 5
  """The clean run is rolled out this many times. MuJoCo Warp on the GPU is not bit-deterministic
  (mujoco_warp issue 562, noted in mjlab's seed_rng): identical runs drift apart after a few seconds,
  so the clean-run numbers are reported as repeat 0 plus the spread over the repeats."""
  n_random_episodes: int = 50
  random_seed_base: int = 1000
  """Randomized episode i uses seed random_seed_base + i (torch, numpy, python, warp)."""
  random_obs_noise: bool = True
  """Randomized runs keep the actor's observation noise (training: enable_corruption=True).
  mjlab's own tracking evaluate.py does the same. The clean run turns it off."""
  push_in_randomized: bool = False
  """Pushes (training: interval 1-3 s, velocity kicks) are NOT part of the 'randomized' protocol;
  they are a separate second-priority run (see push_episodes)."""
  push_episodes: int = 50
  friction_values: tuple[float, ...] = (0.3, 0.6, 1.2)
  """Clean runs at fixed foot friction (training DR range is 0.3-1.2, nominal geom value 0.6)."""

  # ---- tracking error -------------------------------------------------------------------------
  pos_error_thresholds_cm: tuple[float, ...] = (5.0, 10.0, 15.0)
  """An episode 'stays under X' when the body-mean LOCAL position error is <= X cm on every frame
  of the whole clip (an episode that terminates early does not count as under any threshold).
  Chosen for this evaluation, not taken from a paper; for scale, the training terminations fire
  at 25 cm of height error on the torso, ankles or wrists."""

  # ---- smoothness / effort --------------------------------------------------------------------
  joint_limit_margin_deg: float = 3.0
  """'Near a joint position limit' = within this many degrees of the hard MuJoCo joint range.
  Same 3 deg margin as g1/eval_g1_motion.py, so retarget and policy reports agree."""
  velocity_limit_margin_frac: float = 0.10
  """'Near a velocity limit' = |qd| >= (1 - margin) * motor velocity limit."""
  torque_limit_frac: float = 0.99
  """'At the torque limit' = |actuator force| >= frac * actuator forcerange (MuJoCo clips at
  forcerange, so 0.99 catches saturated substeps without float noise)."""

  # ---- foot slip ------------------------------------------------------------------------------
  slip_speed_threshold_cm_s: float = 10.0
  """Report the fraction of contact frames with foot speed above this. Same 10 cm/s as
  g1/eval_g1_motion.py's 'frac_skating_gt_10cm_s'."""

  # ---- beat alignment -------------------------------------------------------------------------
  beat_sigma_s: float = 0.05
  """Gaussian width of the beat alignment score. AIST++/FACT and Bailando use sigma = 3 frames at
  60 fps = 0.05 s; converted to frames at the motion's fps (2.5 frames at 50 fps)."""
  kinetic_smooth_sigma_s: float = 5.0 / 60.0
  """Gaussian smoothing of the kinetic velocity before finding its minima. Bailando
  (utils/beat_align_score.py, used by EDGE's paper) applies gaussian_filter(sigma=5) at 60 fps."""
  music_hop_length: int = 512
  """librosa hop; the sample rate is fps * hop so one librosa frame == one motion frame
  (MINT tools/preprocessing.py: SR = FPS * HOP_LENGTH, HOP_LENGTH = 512)."""
  music_beat_tightness: float = 100.0
  """librosa.beat.beat_track tightness, as in MINT tools/preprocessing.py."""
  calibration_songs: tuple[str, ...] = ("librosa:vibeace", "librosa:fishin", "librosa:nutcracker",
                                        "self:offset=0.0")
  """Wrong-song controls. 'librosa:<key>' = librosa example track (downloaded once to librosa's
  cache), first clip-length seconds. 'self:offset=S' = the right song but from S seconds instead of
  the true start (same tempo and instruments, wrong alignment)."""

  # ---- lag ------------------------------------------------------------------------------------
  max_lag_s: float = 0.5
  """Cross-correlation search window for the reference->executed lag."""

  # ---- per-checkpoint sweep -------------------------------------------------------------------
  sweep_all_checkpoints: bool = True

  # ---- video ----------------------------------------------------------------------------------
  video_width: int = 960
  video_height: int = 720

  # ---- bodies ---------------------------------------------------------------------------------
  body_groups: dict = field(default_factory=lambda: {
    "legs": ["left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
             "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"],
    "arms": ["left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
             "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
    "torso": ["pelvis", "torso_link"],
  })
  feet: dict = field(default_factory=lambda: {"left": "left_ankle_roll_link",
                                              "right": "right_ankle_roll_link"})

  def to_dict(self) -> dict:
    return asdict(self)


# Motor velocity limits (rad/s) from mjlab asset_zoo/robots/unitree_g1/g1_constants.py
# (ElectricActuator.velocity_limit). mjlab does not enforce them in simulation. Waist pitch/roll
# and ankles are driven through a parallel linkage of two 5020 motors; mjlab models them at a
# nominal 1:1 ratio (doubling effort and armature), so the 5020's 37 rad/s is used for them.
MOTOR_VELOCITY_LIMITS = [
  (r".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|"
   r".*_wrist_roll_joint", 37.0, "5020"),
  (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", 32.0, "7520-14"),
  (r".*_hip_roll_joint|.*_knee_joint", 20.0, "7520-22"),
  (r".*_wrist_pitch_joint|.*_wrist_yaw_joint", 22.0, "4010"),
  (r"waist_pitch_joint|waist_roll_joint", 37.0, "2x5020 linkage (1:1 nominal)"),
  (r".*_ankle_pitch_joint|.*_ankle_roll_joint", 37.0, "2x5020 linkage (1:1 nominal)"),
]
