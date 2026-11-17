"""Deterministic all-arm IK and conservative capsule swept-volume preflight.

Capsules must be commissioned enclosing geometry, including the gripper at its
maximum opening. They are hash-pinned model assets, not inferred from URDF names.
"""
from __future__ import annotations

from itertools import combinations
import math
import xml.etree.ElementTree as ET
from typing import Callable, Literal

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from uni_robot_agent.services.environment.contracts import Contract, FiniteFloat
from uni_robot_agent.integrations.piper_x.body.core.contract import DeviceError, ErrorCode
from uni_robot_agent.integrations.piper_x.body.core.arm.tracking import TrackingLimits
from uni_robot_agent.integrations.piper_x.environment.kinematics import from_eef, distance
from ..config import FileRef
from .path import ControlSitePath

IK_REFINE_JOINT_RAD = .15
MAX_REFINEMENT_DEPTH = 5
MAX_PREFLIGHT_SAMPLES = 512
ENTRY_FEEDBACK_TOLERANCE_RAD = .02
ENTRY_COMMAND_MARGIN_RAD = .03


class Capsule(Contract):
    link: str
    start: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    end: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    radius_m: FiniteFloat = Field(gt=0)


class MeshAsset(Contract):
    link: str
    collision_index: int = Field(default=0, ge=0)
    file: FileRef


class CollisionModel(Contract):
    urdf_sha256: str
    capsules: list[Capsule] = Field(min_length=1)
    ignored_self_pairs: list[tuple[str, str]]
    table_exempt_links: list[str]
    enclosure_verified: bool
    scope: Literal["installed_robot", "urdf_only"] = "installed_robot"
    meshes: list[MeshAsset] = Field(default_factory=list)


def segment_distance(a, b, c, d):
    """Distance between finite segments, including parallel and zero-length ones."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
    denom = aa * cc - bb * bb
    candidates = []
    if aa > 1e-16:
        for t in (0., 1.):
            s = np.clip((bb * t - dd) / aa, 0, 1)
            candidates.append(np.linalg.norm(w + s * u - t * v))
    if cc > 1e-16:
        for s in (0., 1.):
            t = np.clip((bb * s + ee) / cc, 0, 1)
            candidates.append(np.linalg.norm(w + s * u - t * v))
    if denom > 1e-16:
        s, t = (bb * ee - cc * dd) / denom, (aa * ee - bb * dd) / denom
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w + s * u - t * v))
    return float(min(candidates, default=np.linalg.norm(w)))


class RobotGeometry:
    def __init__(self, installation):
        self.installation = installation
        self.tracking_limits = TrackingLimits()
        self.model = installation.model
        self.collision_enabled = installation.robot.motion_validation == "collision"
        if not self.collision_enabled and not installation.calibration.operator_supervised:
            raise ValueError("Command-limit execution requires operator supervision")
        self.collision = (CollisionModel.model_validate_json(installation.collision.read_text())
                          if self.collision_enabled else None)
        if self.collision and self.collision.scope == "urdf_only" and not (
                installation.calibration.operator_supervised
                and installation.calibration.geometry_limitations_accepted.strip()):
            raise ValueError("URDF-only geometry requires explicit operator acceptance of omitted tooling")
        root = ET.fromstring(installation.urdf.read_bytes())
        self.joints = list(root.findall("joint"))
        self.links = {link.attrib["name"] for link in root.findall("link")}
        geometry_links = {link.attrib["name"] for link in root.findall("link") if link.findall("collision")}
        covered = {capsule.link for capsule in self.collision.capsules} if self.collision else set()
        if self.collision and (not self.collision.enclosure_verified or self.collision.urdf_sha256 != self.model.urdf.sha256
                or not geometry_links <= covered or not covered <= self.links):
            raise ValueError("Collision model is unverified, incomplete or belongs to another URDF")
        if self.collision and set(self.collision.table_exempt_links) - {self.model.root_link}:
            raise ValueError("Only the fixed base can be exempted from the table check")
        adjacent = {frozenset((j.find("parent").get("link"), j.find("child").get("link"))) for j in self.joints}
        self.ignored = {frozenset(pair) for pair in self.collision.ignored_self_pairs} if self.collision else set()
        if not self.ignored <= adjacent:
            raise ValueError("Only adjacent URDF links may be excluded from self collision")
        for joint in self.joints:
            if joint.get("type") not in {"fixed", "revolute", "continuous"} or joint.find("mimic") is not None:
                raise ValueError("Collision URDF requires independent rotational or fixed joints")
            if joint.get("name") in self.model.joint_names and joint.get("type") == "revolute":
                index = self.model.joint_names.index(joint.get("name"))
                limit = joint.find("limit")
                if (limit is None or self.model.joint_lower[index] < float(limit.get("lower"))
                        or self.model.joint_upper[index] > float(limit.get("upper"))):
                    raise ValueError("Model limits exceed the pinned URDF limits")
        self._origins = {j.get("name"): self._origin(j) for j in self.joints}
        self._axes = {}
        for joint in self.joints:
            if joint.get("type") == "fixed":
                continue
            axis = joint.find("axis")
            direction = np.array([float(v) for v in ("1 0 0" if axis is None else axis.get("xyz")).split()])
            if direction.shape != (3,) or np.linalg.norm(direction) < 1e-12:
                raise ValueError("Invalid URDF axis")
            self._axes[joint.get("name")] = direction / np.linalg.norm(direction)
        by_child = {j.find("child").get("link"): j for j in self.joints}
        self._chain = []
        link = self.model.tip_link
        while link != self.model.root_link:
            if link not in by_child or by_child[link] in self._chain:
                raise ValueError("Disconnected or cyclic URDF")
            joint = by_child[link]
            self._chain.append(joint)
            link = joint.find("parent").get("link")
        self._chain.reverse()
        if [j.get("name") for j in self._chain if j.get("type") != "fixed"] != self.model.joint_names:
            raise ValueError("URDF chain differs from controller joint order")
        # Bounds displacement of any enclosed point per radian for every joint.
        extent = (max(max(np.linalg.norm(c.start), np.linalg.norm(c.end)) + c.radius_m
                      for c in self.collision.capsules) if self.collision else
                  max(c.rotation_radius_m for c in installation.calibration.arms.values()))
        self.reach = sum(np.linalg.norm(self._origin(j)[:3, 3]) for j in self.joints) + extent
        if self.reach <= 0:
            raise ValueError("Degenerate collision geometry")
        self.mesh_checks = None
        if self.collision and self.collision.meshes:
            from .mesh_collision import MeshChecks
            self.mesh_checks = MeshChecks(self, root, self.collision.meshes)

    @staticmethod
    def _origin(joint):
        element = joint.find("origin")
        values = {} if element is None else element.attrib
        return from_eef([float(x) for x in values.get("xyz", "0 0 0").split()],
                        [float(x) for x in values.get("rpy", "0 0 0").split()])

    def links_at(self, name, q):
        if len(q) != 6 or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles")
        c = self.installation.calibration.arms[name]
        frames = {self.model.root_link: np.asarray(c.T_base_from_urdf)}
        values = dict(zip(self.model.joint_names, np.asarray(q) + c.kinematic_joint_offsets_rad))
        pending = list(self.joints)
        while pending:
            progress = False
            for joint in list(pending):
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                if parent not in frames:
                    continue
                t = frames[parent] @ self._origins[joint.get("name")]
                if joint.get("type") != "fixed":
                    if joint.get("name") not in values:
                        raise ValueError("URDF has an unmodelled movable joint")
                    axis = joint.find("axis")
                    direction = np.array([float(v) for v in ("1 0 0" if axis is None else axis.get("xyz")).split()])
                    if np.linalg.norm(direction) < 1e-12:
                        raise ValueError("Invalid URDF axis")
                    turn = np.eye(4)
                    turn[:3, :3] = Rotation.from_rotvec(direction / np.linalg.norm(direction) * values[joint.get("name")]).as_matrix()
                    t = t @ turn
                frames[child] = t
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("Disconnected or cyclic URDF")
        return frames

    def fk_jacobian(self, name, q):
        """Cached serial-chain FK and spatial geometric Jacobian in arm-base axes."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles")
        calibration = self.installation.calibration.arms[name]
        values = q + calibration.kinematic_joint_offsets_rad
        frame = np.asarray(calibration.T_base_from_urdf).copy()
        axes, origins = [], []
        for joint in self._chain:
            frame = frame @ self._origins[joint.get("name")]
            if joint.get("type") != "fixed":
                axis = self._axes[joint.get("name")]
                origins.append(frame[:3, 3].copy())
                axes.append(frame[:3, :3] @ axis)
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * values[len(axes) - 1]).as_matrix()
                frame = frame @ turn
        frame = frame @ np.asarray(calibration.T_tip_from_flange)
        jacobian = np.vstack((np.cross(np.array(axes), frame[:3, 3] - origins).T, np.array(axes).T))
        return frame, jacobian

    def fk(self, name, q):
        return self.fk_jacobian(name, q)[0]

    @staticmethod
    def _rotation_error_jacobian(vector):
        """SO(3) right-Jacobian inverse for target * current.transpose()."""
        x, y, z = vector
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = np.linalg.norm(vector)
        coefficient = (1 / 12 + angle**2 / 720 if angle < 1e-4 else
                       (1 - .5 * angle / math.tan(.5 * angle)) / angle**2)
        return np.eye(3) + .5 * skew + coefficient * skew @ skew

    def ik(self, name, target, initial, *, checkpoint: Callable[[], None] | None = None):
        calibration = self.installation.calibration.arms[name]
        lower, upper = np.asarray(calibration.joint_lower), np.asarray(calibration.joint_upper)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (6,) or not np.isfinite(initial).all() or not np.isfinite(target).all():
            raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "IK requires finite joints and target", stage="preflight")

        def evaluate(q):
            if checkpoint is not None:
                checkpoint()
            actual, jacobian = self.fk_jacobian(name, q)
            rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            residual = np.r_[target[:3, 3] - actual[:3, 3], rotation * .15]
            derivative = np.vstack((-jacobian[:3], -.15 * self._rotation_error_jacobian(rotation) @ jacobian[3:]))
            return residual, derivative

        seed = np.clip(initial, lower, upper)
        residual, _ = evaluate(seed)
        if np.linalg.norm(residual[:3]) <= .001 and np.linalg.norm(residual[3:]) <= .003:
            return seed
        solution = least_squares(lambda q: evaluate(q)[0], seed, jac=lambda q: evaluate(q)[1],
                                 bounds=(lower, upper), max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual, _ = evaluate(solution.x)
        position, rotation = float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]) / .15)
        # Keep the 1 mm seed fast path; allow up to 2 mm after bounded optimization.
        if position <= .002 + 1e-12 and rotation <= .02:
            return solution.x
        margins = np.minimum(solution.x - lower, upper - solution.x)
        raise DeviceError(ErrorCode.LIMIT_VIOLATION, f"{name}: IK failed before dispatch", stage="preflight",
                          details={"reason": "ik_no_verified_solution", "arm": name, "solver": "bounded_trust_region",
                                   "position_residual_m": position, "rotation_residual_rad": rotation,
                                   "position_tolerance_m": .002, "rotation_tolerance_rad": .02,
                                   "evaluations": solution.nfev, "solver_status": int(solution.status),
                                   "candidate_joints_rad": solution.x.tolist(), "joint_margins_rad": margins.tolist(),
                                   "active_joint_limits": [dict(joint_index=i, lower=float(lower[i]), upper=float(upper[i]),
                                                                value=float(solution.x[i]))
                                                           for i in range(6) if margins[i] < 1e-5]})

    def check(self, positions, inflation=None, *, measured=False):
        capsules = []
        cal = self.installation.calibration
        for name, q in positions.items():
            c = cal.arms[name]
            q = np.asarray(q)
            tolerance = ENTRY_FEEDBACK_TOLERANCE_RAD if measured and not self.collision_enabled else 0.
            if (q.shape != (6,) or not np.isfinite(q).all()
                    or np.any(q < np.asarray(c.joint_lower) - tolerance)
                    or np.any(q > np.asarray(c.joint_upper) + tolerance)):
                raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Joint range exceeded", stage="preflight")
        if not self.collision_enabled:
            return
        if self.mesh_checks is not None:
            return self.mesh_checks.check(positions, inflation)
        for name, q in positions.items():
            c = cal.arms[name]
            if np.any(np.asarray(q) < c.joint_lower) or np.any(np.asarray(q) > c.joint_upper):
                raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Joint range exceeded", stage="preflight")
            frames = self.links_at(name, q)
            for shape in self.collision.capsules:
                t = np.asarray(c.T_world_from_base) @ frames[shape.link]
                a, b = [(t @ np.r_[p, 1])[:3] for p in (shape.start, shape.end)]
                # Include the commissioned tracking tube around the planned state.
                # This is conservative even when joints do not advance synchronously.
                radius = shape.radius_m + (inflation or {}).get(name, 0.)
                if shape.link not in self.collision.table_exempt_links and min(a[2], b[2]) - radius <= cal.table_height_m + cal.collision_margin_m:
                    raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Path intersects table clearance", stage="preflight")
                capsules.append((name, shape.link, a, b, radius))
        for a, b in combinations(capsules, 2):
            if a[0] == b[0] and (a[1] == b[1] or frozenset((a[1], b[1])) in self.ignored):
                continue
            if segment_distance(a[2], a[3], b[2], b[3]) <= a[4] + b[4] + cal.collision_margin_m:
                raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Self or inter-arm collision clearance failed", stage="preflight")

    def goal_check(self, name, target, seeds, *, checkpoint=None):
        """Endpoint evidence for supplied local seeds; path admission stays separate."""
        attempts = []
        for label, seed in seeds:
            try:
                joints = self.ik(name, target, seed, checkpoint=checkpoint)
            except DeviceError as error:
                if (error.details or {}).get("reason") != "ik_no_verified_solution":
                    raise
                attempts.append({"seed": label, **error.details})
                continue
            return {"status": "verified_endpoint", "seed": label, "joints_rad": joints.tolist(),
                    "path_checked": False, "ik_branches": "supplied_local_seeds"}
        return {"status": "no_verified_solution", "attempts": attempts,
                "path_checked": False, "ik_branches": "supplied_local_seeds"}

    def _within_command_step(self, start, end):
        for name in start:
            limits = self.installation.calibration.arms[name].motion_limits
            if np.max(np.abs(end[name] - start[name])) > limits.joint_step_rad:
                return False
            dp, angle = distance(self.fk(name, start[name]), self.fk(name, end[name]))
            if dp > limits.eef_step_m or angle > limits.eef_rotation_step_rad:
                return False
        return True

    def command_step_evidence(self, start, end):
        evidence = {}
        for name in start:
            limits = self.installation.calibration.arms[name].motion_limits
            position, angle = distance(self.fk(name, start[name]), self.fk(name, end[name]))
            evidence[name] = {"measured_joints_rad": np.asarray(start[name]).tolist(),
                              "target_joints_rad": np.asarray(end[name]).tolist(),
                              "max_joint_delta_rad": float(np.max(np.abs(end[name] - start[name]))),
                              "joint_delta_limit_rad": limits.joint_step_rad,
                              "position_delta_m": position, "position_limit_m": limits.eef_step_m,
                              "rotation_delta_rad": angle, "rotation_limit_rad": limits.eef_rotation_step_rad}
        return evidence

    def _subdivisions(self, start, end):
        if self.collision_enabled:
            return max(1, *(math.ceil(self.reach * np.sum(np.abs(end[n] - start[n])) /
                            (self.installation.calibration.collision_margin_m / 4)) for n in start))
        # Retain fine kinematic sampling and command-step bounds without using
        # scene margins or implying any geometric collision clearance.
        counts = [1]
        for name in start:
            limits = self.installation.calibration.arms[name].motion_limits
            delta = np.abs(end[name] - start[name])
            counts.extend((math.ceil(np.max(delta) / (limits.joint_step_rad / 4)),
                           math.ceil(self.reach * np.sum(delta) / (limits.eef_step_m / 4)),
                           math.ceil(np.sum(delta) / (limits.eef_rotation_step_rad / 4))))
        return max(counts)

    def check_measured_segment(self, start, end, *, checkpoint=None):
        """Validate the actual start to the next target, not an assumed arrival.

        Collision mode repeats the preflight swept-volume proof. Command-limit
        mode retains native range and step checks on the measured connection.
        """
        self.check(start, measured=not self.collision_enabled)
        self.check(end)
        if not self._within_command_step(start, end):
            return False
        if not self.collision_enabled:
            # Each native command endpoint is bounded. The only interval outside
            # the bounds is the monotone entry from near-bound measured feedback.
            return True
        steps = self._subdivisions(start, end)
        if steps > 2000:
            raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Measured segment subdivision budget exceeded")
        for i in range(1, steps + 1):
            if checkpoint is not None:
                checkpoint()
            row = {n: start[n] + (end[n] - start[n]) * i / steps for n in start}
            inflation = {n: (np.abs(end[n] - start[n]) / steps +
                        self.installation.calibration.arms[n].tracking_tolerance_rad
                        if self.mesh_checks is not None else
                        self.reach * (np.sum(np.abs(end[n] - start[n])) / steps +
                        6 * self.installation.calibration.arms[n].tracking_tolerance_rad)) for n in start}
            self.check(row, inflation)
        return True

    def entry_recovery(self, measured):
        """Return one bounded entry destination, never the original task goal."""
        if self.collision_enabled:
            return None
        self.check(measured, measured=True)
        start = {name: np.asarray(q, dtype=float) for name, q in measured.items()}
        goal = {name: q.copy() for name, q in start.items()}
        changed = False
        for name, q in start.items():
            calibration = self.installation.calibration.arms[name]
            lower, upper = np.asarray(calibration.joint_lower), np.asarray(calibration.joint_upper)
            margin = np.minimum(ENTRY_COMMAND_MARGIN_RAD, (upper - lower) / 4)
            below, above = q < lower, q > upper
            if not np.any(below | above):
                continue
            # An exact-bound peer joint can drift outside while this arm enters.
            # Give its command the same margin without moving an unaffected arm.
            near_lower, near_upper = q < lower + margin, q > upper - margin
            goal[name][near_lower] = (lower + margin)[near_lower]
            goal[name][near_upper] = (upper - margin)[near_upper]
            changed = True
        if not changed:
            return None
        self.check(goal)
        if not self._within_command_step(start, goal):
            raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Joint entry exceeds command step limits", stage="preflight")
        return goal

    def plan(self, measured, targets, *, segment_m, max_segments,
             checkpoint: Callable[[], None] | None = None):
        """Check the complete joint path, including non-participating measured holds."""
        if measured.keys() != targets.keys():
            raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "A path must name every arm")
        current = {n: np.array(q, dtype=float) for n, q in measured.items()}
        starts = {n: self.fk(n, q) for n, q in current.items()}
        goals, site_paths, segments = {}, {}, 1
        for name, target in targets.items():
            if "hold_joints" in target:
                tolerance = self.tracking_limits.hold_admission_joint_rad
                if not np.allclose(target["hold_joints"], current[name], atol=tolerance, rtol=0):
                    raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Held joints differ from measured state")
                goals[name] = starts[name]
            else:
                goals[name] = from_eef(target["target_position_m"], target["target_rpy_rad"])
                site_paths[name] = ControlSitePath(starts[name], goals[name], target.get("control_offset_m", [0., 0., 0.]))
                radius = self.installation.calibration.arms[name].rotation_radius_m
                segments = max(segments, math.ceil(site_paths[name].work(radius) / segment_m))
        if segments > max_segments:
            raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Path exceeds the requested travel budget")
        if checkpoint is not None:
            checkpoint()
        self.check(current)
        path = [current]
        samples = 0

        def solve_interval(previous_fraction, fraction, depth=0):
            nonlocal current, samples
            if checkpoint is not None:
                checkpoint()
            samples += 1
            if samples > MAX_PREFLIGHT_SAMPLES:
                raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Path refinement budget exceeded", stage="preflight")
            next_q = {}
            try:
                for name in current:
                    if name in site_paths:
                        next_q[name] = self.ik(name, site_paths[name].at(fraction), current[name], checkpoint=checkpoint)
                    else:
                        next_q[name] = current[name].copy()
            except DeviceError as error:
                error.details = {**(error.details or {}), "path_fraction": fraction,
                                 "previous_path_fraction": previous_fraction, "refinement_depth": depth}
                if error.details.get("reason") == "ik_no_verified_solution" and depth < MAX_REFINEMENT_DEPTH:
                    middle = (previous_fraction + fraction) / 2
                    solve_interval(previous_fraction, middle, depth + 1)
                    solve_interval(middle, fraction, depth + 1)
                    return
                raise
            largest_jump = max(float(np.max(np.abs(next_q[n] - current[n]))) for n in current)
            if largest_jump > IK_REFINE_JOINT_RAD:
                if depth >= MAX_REFINEMENT_DEPTH:
                    raise DeviceError(ErrorCode.LIMIT_VIOLATION, "IK path continuity could not be verified", stage="preflight",
                                      details={"reason": "ik_branch_discontinuity", "path_fraction": fraction,
                                               "max_joint_delta_rad": largest_jump, "refinement_depth": depth})
                middle = (previous_fraction + fraction) / 2
                solve_interval(previous_fraction, middle, depth + 1)
                solve_interval(middle, fraction, depth + 1)
                return
            # Cover each joint interpolation interval with inflated endpoint capsules.
            # The reach bound covers all joints, making this conservative between samples.
            steps = self._subdivisions(current, next_q)
            if steps > 2000:
                raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Path subdivision budget exceeded")
            command_start, previous = current, current
            for i in range(1, steps + 1):
                if checkpoint is not None:
                    checkpoint()
                row = {n: current[n] + (next_q[n] - current[n]) * i / steps for n in current}
                if self.mesh_checks is not None:
                    inflation = {n: np.abs(next_q[n] - current[n]) / steps +
                        self.installation.calibration.arms[n].tracking_tolerance_rad for n in current}
                else:
                    inflation = {n: self.reach * (np.sum(np.abs(next_q[n] - current[n])) / steps +
                        6 * self.installation.calibration.arms[n].tracking_tolerance_rad) for n in current}
                self.check(row, inflation)
                # These are collision proof samples on one linear joint interval.
                # Merge only collinear samples; never change the checked joint path
                # by taking a shortcut across two IK waypoints.
                if not self._within_command_step(command_start, row):
                    if previous is command_start or not self._within_command_step(previous, row):
                        raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Collision sample exceeds command step limits")
                    path.append(previous)
                    command_start = previous
                previous = row
            path.append(next_q)
            current = next_q

        try:
            for index in range(1, segments + 1):
                solve_interval((index - 1) / segments, index / segments)
        except DeviceError as error:
            name = (error.details or {}).get("arm")
            if name in site_paths:
                error.details["goal_joint_check"] = self.goal_check(name, goals[name],
                    [("last_verified", current[name]), ("measured_start", measured[name])], checkpoint=checkpoint)
            raise
        return path
