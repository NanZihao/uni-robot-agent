"""Registered control-site interpolation, following the ARX ONE path contract."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..environment.kinematics import bounded_target, distance
from ..body.core.arm.motion_limits import QUINTIC_PEAK_SPEED, QUINTIC_PEAK_ACCELERATION

if TYPE_CHECKING:
    from .planning import RobotGeometry


class JointTrajectory:
    """Monotone joint interpolation with one bounded, rest-to-rest time base.

    Every joint stays between its adjacent checked endpoints. Callers must
    preflight the interpolated curve's swept bounds before dispatch.
    """

    def __init__(self, path: list[dict[str, np.ndarray]], geometry: RobotGeometry,
                 minimum_duration: float):
        self.names = list(path[0])
        values = np.array([[row[name] for name in self.names] for row in path])
        lengths = np.max(np.abs(np.diff(values, axis=0)), axis=(1, 2))
        keep = np.r_[True, lengths > 1e-12]
        values = values[keep]
        if len(values) == 1:
            values = np.repeat(values, 2, axis=0)
            lengths = np.ones(1)
        else:
            lengths = np.max(np.abs(np.diff(values, axis=0)), axis=(1, 2))
        self.knots = np.r_[0., np.cumsum(lengths) / sum(lengths)]
        self.curve = PchipInterpolator(self.knots, values, axis=0)
        period = 1 / geometry.installation.robot.control_frequency_hz
        self.duration = max(minimum_duration, period)
        # Bound derivatives analytically on every cubic interval, including
        # interior velocity extrema and both one-sided accelerations at knots.
        a, b, c, _ = self.curve.c
        widths = np.diff(self.knots)[:, None, None]
        vertex = np.clip(np.divide(-b, 3 * a, out=np.zeros_like(a), where=a != 0), 0, widths)
        speed = np.maximum.reduce([np.abs(c), np.abs(3*a*widths**2 + 2*b*widths + c),
                                   np.abs(3*a*vertex**2 + 2*b*vertex + c)])
        acceleration = np.maximum(np.abs(2*b), np.abs(6*a*widths + 2*b))
        speed *= QUINTIC_PEAK_SPEED
        acceleration = acceleration * QUINTIC_PEAK_SPEED**2 + speed / QUINTIC_PEAK_SPEED * QUINTIC_PEAK_ACCELERATION
        for index, name in enumerate(self.names):
            calibration = geometry.installation.calibration.arms[name]
            limits = calibration.motion_limits
            velocity = speed[:, index]
            angular_speed = velocity.sum(axis=1)
            angular_acceleration = acceleration[:, index].sum(axis=1)
            radius = (geometry.reach
                      + np.linalg.norm(np.asarray(calibration.T_tip_from_flange)[:3, 3])
                      + np.linalg.norm(np.asarray(calibration.T_flange_from_grasp)[:3, 3]))
            # Serial-chain bounds are |p_dot| <= R*sum(|q_dot|) and
            # |p_ddot| <= R*(sum(|q_ddot|) + sum(|q_dot|)**2).
            self.duration = max(self.duration,
                float(velocity.max()) / limits.joint_speed_rad_s,
                float(angular_speed.max()) / limits.eef_rotation_speed_rad_s,
                float(radius * angular_speed.max()) / limits.eef_speed_m_s,
                period * float(velocity.max()) / limits.joint_step_rad,
                period * float(angular_speed.max()) / limits.eef_rotation_step_rad,
                period * float(radius * angular_speed.max()) / limits.eef_step_m,
                math.sqrt(float((radius * (angular_acceleration + angular_speed**2)).max()) / limits.eef_accel_m_s2))

    def at_parameter(self, parameter: float) -> dict[str, np.ndarray]:
        values = self.curve(parameter)
        return {name: values[index].copy() for index, name in enumerate(self.names)}

    def at(self, fraction: float) -> dict[str, np.ndarray]:
        blend = fraction**3 * (10 + fraction * (-15 + 6*fraction))
        return self.at_parameter(blend)


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
