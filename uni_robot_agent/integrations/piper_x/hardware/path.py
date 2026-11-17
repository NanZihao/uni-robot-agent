"""Registered control-site interpolation, following the ARX ONE path contract."""
from __future__ import annotations

import math

import numpy as np

from ..environment.kinematics import bounded_target, distance


class ControlSitePath:
    """Interpolate site translation and orientation, then recover flange poses."""

    def __init__(self, start: np.ndarray, goal: np.ndarray, offset=(0., 0., 0.)):
        self.start, self.goal = start.copy(), goal.copy()
        self.offset = np.asarray(offset, dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError("Expected three finite control-offset values")
        self.first = start[:3, 3] + start[:3, :3] @ self.offset
        self.last = goal[:3, 3] + goal[:3, :3] @ self.offset
        self.translation, self.angle = distance(start, goal)
        self.site_distance = float(np.linalg.norm(self.last - self.first))
        self.radius = float(np.linalg.norm(self.offset))

    def work(self, rotation_radius: float) -> float:
        return self.site_distance + max(rotation_radius, self.radius) * self.angle

    def at(self, fraction: float) -> np.ndarray:
        if not 0 <= fraction <= 1:
            raise ValueError("Path fraction must lie in [0, 1]")
        if fraction == 0:
            return self.start.copy()
        if fraction == 1:
            return self.goal.copy()
        pose = bounded_target(self.start, self.goal,
                              translation=max(self.translation, 1e-12) * fraction,
                              rotation=max(self.angle, 1e-12) * fraction)
        pose[:3, 3] = self.first + fraction * (self.last - self.first) - pose[:3, :3] @ self.offset
        return pose

    def minimum_duration(self, limits) -> float:
        length = self.site_distance + self.radius * self.angle
        return max(limits.eef_duration_s(length, self.angle), math.sqrt(
            ((10 / math.sqrt(3)) * length + 1.875**2 * self.radius * self.angle**2)
            / limits.eef_accel_m_s2))
