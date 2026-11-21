"""Minimal pure-Python stand-in for pytorch3d.transforms (EDGE only needs the
rotation conversions, plus RotateAxisAngle for dataset preprocessing). The
conversions are pytorch3d's own rotation_conversions.py, BSD-licensed."""
import math

import torch

from .rotation_conversions import *  # noqa: F401,F403
from .rotation_conversions import _axis_angle_rotation


class RotateAxisAngle:
  """Subset of pytorch3d.transforms.RotateAxisAngle: rotate points about X/Y/Z."""

  def __init__(self, angle, axis="X", degrees=True, dtype=torch.float32, device="cpu"):
    angle = torch.as_tensor(angle, dtype=dtype, device=device)
    if degrees:
      angle = angle * math.pi / 180.0
    # pytorch3d applies points as row vectors: p @ R^T with R from _axis_angle_rotation
    self._matrix = _axis_angle_rotation(axis.upper(), angle)

  def transform_points(self, points):
    R = self._matrix.to(points)
    return points @ R.transpose(-1, -2)
