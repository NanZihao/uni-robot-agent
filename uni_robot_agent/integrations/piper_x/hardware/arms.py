"""One robot's arms share admission, preflight and stop scope in one Body."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import math
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from uni_robot_agent.integrations.piper_x.body.core.arm.eef import coordinate_metadata
from uni_robot_agent.integrations.piper_x.body.core.arm.arrival import ArrivalWindow
from uni_robot_agent.integrations.piper_x.body.core.arm.tracking import PoseTrackingGuard
from uni_robot_agent.integrations.piper_x.body.core.contract import DeviceError, ErrorCode
from uni_robot_agent.integrations.piper_x.body.core.manifest import BodyManifest, DeviceManifest, Signal, Procedure, ObservationStreamSpec, num
from uni_robot_agent.integrations.piper_x.body.core.observations import ObservationBuffer
from uni_robot_agent.integrations.piper_x.body.core.resources import ResourceLocks
from ..config import Installation
from .identity import verify_can
from .planning import RobotGeometry
from .path import ControlSitePath, JointTrajectory
from .transport import PassiveCan, SdkWriter
from uni_robot_agent.integrations.piper_x.environment.kinematics import Kinematics, distance, from_eef


class ArmBody:
    data_origin = "physical"

    def __init__(self, deployment, *, body_id, execute=False):
        self.installation = Installation(deployment)
        self.robot = self.installation.robot
        self.body_id = body_id
        self.execute = execute
        self._cleanup = ExitStack()
        self._blocked = threading.Event()
        self._planning_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stop_receipt = None
        self._resume_stop_id: str | None = None
        self.tracking_limits = self.robot.tracking_limits
        self._hold_references: dict[str, dict] = {}
        self._readers, self._writers = {}, {}
        self._by_device = {a.device_id: name for name, a in self.robot.arms.items()}
        self._histories = {d: ObservationBuffer() for d in self._by_device}
        self._last_stamp = {}
        self._history_lock = threading.Lock()
        self._closed = threading.Event()
        if self.installation.host.bodies[self.robot.arms_body].body_id != body_id:
            raise ValueError("Wrong arm Body identity")
        if execute:
            self.installation.require_motion()
        try:
            resources = [verify_can(self.installation.host.arms[a.binding]) for a in self.robot.arms.values()]
            locks = ResourceLocks(resources, self.installation.host.lock_dir)
            self._cleanup.callback(locks.close)
            self.geometry = RobotGeometry(self.installation) if execute else None
            environment = self.installation.environment()
            self._kinematics = {name: Kinematics(environment.arms[name]) for name, calibration
                                in self.installation.calibration.arms.items() if any(calibration.kinematic_joint_offsets_rad)}
            for name, arm in self.robot.arms.items():
                reader = PassiveCan(self.installation.host.arms[arm.binding].interface,
                    max_age_s=self.robot.feedback_max_age_s, max_skew_s=self.robot.feedback_max_skew_s)
                self._cleanup.callback(reader.close)
                self._readers[name] = reader
            self._history_thread = threading.Thread(target=self._collect, daemon=True, name=f"history-{body_id}")
            self._history_thread.start()
            self._cleanup.callback(self._end_history)
            if execute:
                for name in self.robot.arms:
                    deadline = time.monotonic() + self.robot.action_timeout_s
                    while True:
                        try:
                            self._require_feedback(name)
                            break
                        except RuntimeError:
                            if time.monotonic() >= deadline:
                                raise
                            self._closed.wait(.01)
                    writer = SdkWriter(self.installation, name)
                    self._cleanup.callback(writer.close)
                    self._writers[name] = writer
        except BaseException:
            self._cleanup.close()
            raise

    def _collect(self):
        while not self._closed.wait(.005):
            for device, name in self._by_device.items():
                try:
                    data = self._readers[name].read()
                    with self._history_lock:
                        for row in data["joint_history"]:
                            stamp = int(row["timing"]["source_timestamp_ns"])
                            if stamp <= self._last_stamp.get(device, 0):
                                continue
                            self._histories[device].append(row["timing"], {"joint_position_rad": row["joint_position_rad"]},
                                generation=1, sequence=stamp)
                            self._last_stamp[device] = stamp
                except Exception:
                    # No fabricated samples: freshness gates and history queries fail closed.
                    continue

    def _end_history(self):
        self._closed.set()
        self._history_thread.join(timeout=1)

    def manifest(self):
        devices = []
        for name, arm in self.robot.arms.items():
            c = self.installation.calibration.arms[name]
            devices.append(DeviceManifest(arm.device_id, "arm", hardware_profile="piper_x",
                frames=[f"{arm.device_id}/base", f"{arm.device_id}/flange"],
                signals={n: Signal(n) for n in ("joint_position_rad", "eef_position_m", "eef_rpy_rad", "native_eef_position_m", "native_eef_rpy_rad", "gripper_opening", "gripper_opening_measured",
                    "controller_motion_status_code", "controller_target_reached")},
                observation_streams={"state": ObservationStreamSpec(["joint_position_rad"])},
                procedures={
                    "move_eef_group": Procedure("move_eef_group", "Preflight every arm before any dispatch", params={
                        "targets": {"type": "object"}, "max_segments": {"type": "integer", "minimum": 1, "maximum": 24},
                        "segment_m": num(.0001, .06), "max_duration_s": num(.1, 60)},
                        required=["targets", "max_segments", "segment_m", "max_duration_s"], completion="measured_target_arrival"),
                    "resume_control": Procedure("resume_control", "Reopen native admission after an evidenced tracking stop",
                        params={"stop_id": {"type": "string", "minLength": 1}}, required=["stop_id"],
                        completion="native_admission_resumed")} if self.execute else {},
                metadata={"identity": self.installation.identity(name, "arms"),
                    "joint_limits_rad": {"lower": c.joint_lower, "upper": c.joint_upper},
                    "motion_limits": c.motion_limits.model_dump(), "eef_coordinates": coordinate_metadata(arm.device_id),
                    "tracking_limits": self.tracking_limits.to_dict(),
                    "kinematic_joint_offsets_rad": c.kinematic_joint_offsets_rad,
                    "eef_pose_source": "native feedback with calibrated kinematic correction" if any(c.kinematic_joint_offsets_rad) else "native feedback",
                    "motion_validation": self.robot.motion_validation,
                    "collision_checking_enabled": self.robot.motion_validation == "collision",
                    "model_sha256": self.installation.model.urdf.sha256, "motion_enabled": self.execute},
                limitations=["Host CAN receive timestamps; no controller hardware clock", "Software stop is not a certified E-stop",
                    *(["Command-limit validation only; no table, self or inter-arm collision checks"]
                      if self.robot.motion_validation == "command_limits" else []),
                    *(["Collision checks cover the URDF only; omitted tooling requires operator supervision"]
                      if self.geometry and self.geometry.collision and self.geometry.collision.scope == "urdf_only" else [])]))
        return BodyManifest(self.body_id, "piper_x", devices)

    def read(self, device_id, include=None):
        name = self._by_device[device_id]
        data = self._readers[name].read()
        c = self.installation.calibration.arms[name]
        width = data["agx_gripper"].get("width_m")
        if width is None or not c.gripper_closed_m - .001 <= width <= c.gripper_open_m + .001:
            raise DeviceError(ErrorCode.OBSERVATION_STALE, "Gripper mode or calibration mismatch")
        opening = float(np.clip((width - c.gripper_closed_m) / (c.gripper_open_m - c.gripper_closed_m), 0, 1))
        measured = self._measured_flange(name, data)
        position, rpy = data["flange_position_m"], data["flange_rpy_rad"]
        if any(c.kinematic_joint_offsets_rad):
            position = measured[:3, 3].tolist()
            rpy = Rotation.from_matrix(measured[:3, :3]).as_euler("xyz").tolist()
        return {"signals": {"joint_position_rad": data["joints_rad"], "eef_position_m": position,
                "eef_rpy_rad": rpy, "native_eef_position_m": data["flange_position_m"],
                "native_eef_rpy_rad": data["flange_rpy_rad"], "gripper_opening": opening, "gripper_opening_measured": True,
                "controller_motion_status_code": data.get("controller_motion_status_code"),
                "controller_target_reached": data.get("controller_target_reached")},
                "freshness": "fresh", "age_s": data["age_s"], "moving": data["moving"], "timing": data["timing"],
                "fault": data["fault"], "data_origin": self.data_origin}

    def _require_feedback(self, name):
        data = self._readers[name].read()
        if data["fault"] or not data["can_control"] or not data["joint_mode"]:
            raise DeviceError(ErrorCode.PRECONDITION_FAILED, "Commission fault-free CAN joint mode externally; no automatic mode changes")
        return data

    def check(self, device_id):
        try:
            data = self.read(device_id)
            if data["fault"] or self._blocked.is_set():
                raise RuntimeError("Fault or blocked motion")
            if self.execute:
                self._require_feedback(self._by_device[device_id])
            return {"readiness": "satisfied", "checks": {"feedback": {"state": "pass"}}, "motion_enabled": self.execute}
        except Exception as error:
            return {"readiness": "unsatisfied", "checks": {"feedback": {"state": "fail", "error": str(error)}}}

    def _preflight_feedback(self, names, checkpoint):
        """Wait briefly for coherent fresh feedback before any physical write."""
        deadline = time.monotonic() + min(.1, self.robot.feedback_max_age_s)
        while True:
            checkpoint()
            try:
                return {name: self._require_feedback(name) for name in names}
            except DeviceError as error:
                error.stage = "preflight"
                raise
            except RuntimeError as error:
                if time.monotonic() >= deadline:
                    raise DeviceError(ErrorCode.OBSERVATION_STALE,
                        "Fresh coherent feedback unavailable before dispatch: " + str(error),
                        execution_outcome="none", stage="preflight") from error
                time.sleep(.005)

    def _arrival_error(self, name, data, target_q, *, held, require_open=False):
        """Cartesian requests settle at the grasp site; held arms settle in joints."""
        c = self.installation.calibration.arms[name]
        joint_error = float(np.max(np.abs(np.asarray(data["joints_rad"]) - target_q)))
        evidence = {"max_joint_error_rad": joint_error}
        open_ok = True
        if require_open:
            width = data["agx_gripper"].get("width_m")
            opening = ((width - c.gripper_closed_m) / (c.gripper_open_m - c.gripper_closed_m)
                       if type(width) in (int, float) and math.isfinite(width) else None)
            open_ok = opening is not None and abs(1. - opening) <= self.tracking_limits.settle_gripper_opening
            evidence.update(gripper_opening=opening, gripper_opening_measured=opening is not None,
                            gripper_open_verified=open_ok,
                            gripper_opening_tolerance=self.tracking_limits.settle_gripper_opening)
        if held:
            evidence["joint_tolerance_rad"] = self.tracking_limits.settle_joint_rad
            return open_ok and joint_error <= self.tracking_limits.settle_joint_rad, evidence
        offset = np.asarray(c.T_flange_from_grasp)
        measured = self._measured_flange(name, data) @ offset
        target = self.geometry.fk(name, target_q) @ offset
        position, rotation = distance(measured, target)
        position_limit = self.tracking_limits.settle_position_m
        rotation_limit = self.tracking_limits.settle_rotation_rad
        evidence.update(position_error_m=position, rotation_error_rad=rotation,
                        position_tolerance_m=position_limit, orientation_tolerance_rad=rotation_limit)
        return open_ok and position <= position_limit and rotation <= rotation_limit, evidence

    def _hold_monitors(self, targets: dict, actual: dict) -> dict:
        monitors = {}
        for name, target in targets.items():
            if "hold_joints" in target:
                reference = self._hold_references.setdefault(name, {
                    "joints_rad": list(actual[name]["joints_rad"]),
                    "grasp_pose": (self._measured_flange(name, actual[name]) @
                                   np.asarray(self.installation.calibration.arms[name].T_flange_from_grasp)).tolist(),
                    "basis": "first_measured_hold"})
                monitors[name] = {"reference": reference, "target": np.asarray(target["hold_joints"]),
                                  "peak_error_rad": 0.}
        self._update_holds(monitors, actual)
        return monitors

    def _update_holds(self, monitors: dict, actual: dict) -> None:
        for name, monitor in monitors.items():
            error = float(np.max(np.abs(np.asarray(actual[name]["joints_rad"]) - monitor["target"])))
            monitor["peak_error_rad"] = max(monitor["peak_error_rad"], error)

    def _hold_reports(self, monitors: dict, actual: dict, arrivals: dict) -> dict:
        reports = {}
        for name, monitor in monitors.items():
            data = actual[name]
            joints = np.asarray(data["joints_rad"])
            reference = monitor["reference"]
            pose = self._measured_flange(name, data) @ np.asarray(
                self.installation.calibration.arms[name].T_flange_from_grasp)
            position, rotation = distance(np.asarray(reference["grasp_pose"]), pose)
            cumulative = float(np.max(np.abs(joints - reference["joints_rad"])))
            verified = arrivals.get(name, {}).get("verified", False)
            warning = (monitor["peak_error_rad"] > self.tracking_limits.settle_joint_rad
                       or cumulative > self.tracking_limits.settle_joint_rad
                       or position > self.tracking_limits.settle_position_m
                       or rotation > self.tracking_limits.settle_rotation_rad or not verified)
            reports[name] = {"role": "joint_hold", "reference": reference,
                "command_joint_target_rad": monitor["target"].tolist(), "actual_joints_rad": joints.tolist(),
                "threshold_rad": self.tracking_limits.settle_joint_rad,
                "max_joint_error_rad": monitor["peak_error_rad"],
                "final_joint_error_rad": float(np.max(np.abs(joints - monitor["target"]))),
                "cumulative_joint_error_rad": cumulative, "cumulative_position_error_m": position,
                "cumulative_rotation_error_rad": rotation, "arrival_verified": verified,
                "feedback_timing": data["timing"], "status": "deviation_warning" if warning else "within_tolerance"}
        return reports

    def _measured_flange(self, name, data):
        native = from_eef(data["flange_position_m"], data["flange_rpy_rad"])
        if not any(self.installation.calibration.arms[name].kinematic_joint_offsets_rad):
            return native
        return self._kinematics[name].correct_feedback(data["joints_rad"], native)

    def _tracking_pose(self, name, data):
        if not any(self.installation.calibration.arms[name].kinematic_joint_offsets_rad):
            return [*data["flange_position_m"], *data["flange_rpy_rad"]]
        measured = self._measured_flange(name, data)
        return [*measured[:3, 3], *Rotation.from_matrix(measured[:3, :3]).as_euler("xyz")]

    def _tracking_target(self, name, joints):
        pose = self.geometry.fk(name, joints)
        return [*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")]

    def _check_pose_tracking(self, guards, actual, *, segment):
        for name, data in actual.items():
            stamp = int(data["timing"]["received_monotonic_ns"]) / 1e9
            violation = guards[name].check(stamp, self._tracking_pose(name, data))
            if violation is not None:
                raise DeviceError(ErrorCode.TRACKING_ERROR, "Sustained end-effector tracking deviation",
                    details={"arm": name, "segment": segment, **violation})

    def health(self):
        return {"ok": not self._closed.is_set(), "robot_id": self.robot.robot_id, "motion_enabled": self.execute}

    def read_observation_at(self, device_id, stream, at, *, mode="bracket", max_gap_s=.05, include=None):
        if stream != "state":
            raise DeviceError(ErrorCode.NOT_SUPPORTED, "Unknown observation stream")
        result = self._histories[device_id].query(at, mode=mode, max_gap_s=max_gap_s)
        for sample in result["samples"]:
            sample["signals"] = sample.pop("payload")
        return result

    def invoke(self, device_ids, procedure, params, *, cancel, progress):
        if procedure == "resume_control":
            return self._resume_control(device_ids, params, cancel)
        if not self.execute or self.geometry is None or self._blocked.is_set():
            raise DeviceError(ErrorCode.PRECONDITION_FAILED, "Motion is disabled or blocked")
        if procedure != "move_eef_group" or set(device_ids) != set(self._by_device):
            raise DeviceError(ErrorCode.NOT_SUPPORTED, "Use one coordinated path naming all robot arms")
        if set(params["targets"]) != set(device_ids):
            raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "Targets differ from invoked arms")
        with self._planning_lock:
            deadline = time.monotonic() + params["max_duration_s"]
            planning_deadline = time.monotonic() + min(8., params["max_duration_s"] / 2)

            def checkpoint():
                if cancel.is_set() or self._blocked.is_set():
                    raise DeviceError(ErrorCode.PRECONDITION_FAILED, "Path cancelled before dispatch", stage="preflight")
                if time.monotonic() >= planning_deadline:
                    raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Path planning time budget exceeded before dispatch", stage="preflight")

            checkpoint()
            before = self._preflight_feedback(self.robot.arms, checkpoint)
            measured = {n: d["joints_rad"] for n, d in before.items()}
            targets = {n: dict(params["targets"][a.device_id]) for n, a in self.robot.arms.items()}
            requested_durations = [target.get("duration_s", .2) for target in targets.values()]
            if any(type(value) not in (float, int) or not np.isfinite(value)
                   or not .1 <= value <= params["max_duration_s"] for value in requested_durations):
                raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "Target duration exceeds the action budget", stage="preflight")
            minimum_duration = max(requested_durations)
            travel_limited = False
            for n, target in targets.items():
                if "hold_joints" not in target and target.get("target_frame") != self.robot.arms[n].device_id + "/base":
                    raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "Wrong native target frame")
                opening = target.get("gripper_opening")
                if type(opening) not in (float, int) or not np.isfinite(opening) or not 0 <= opening <= 1:
                    raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "Expected normalized gripper target")
                if "hold_joints" not in target:
                    offset = np.asarray(target.get("control_offset_m", [0., 0., 0.]), dtype=float)
                    if offset.shape != (3,) or not np.isfinite(offset).all():
                        raise DeviceError(ErrorCode.INVALID_ARGUMENTS, "Expected three finite control-offset values", stage="preflight")
                    start = self.geometry.fk(n, measured[n])
                    goal = from_eef(target["target_position_m"], target["target_rpy_rad"])
                    site_path = ControlSitePath(start, goal, target.get("control_offset_m", [0., 0., 0.]))
                    travel = site_path.work(self.installation.calibration.arms[n].rotation_radius_m)
                    budget = params["segment_m"] * params["max_segments"]
                    if travel > budget:
                        fraction = budget * .999 / travel
                        goal = site_path.at(fraction)
                        target["target_position_m"] = goal[:3, 3].tolist()
                        target["target_rpy_rad"] = Rotation.from_matrix(goal[:3, :3]).as_euler("xyz").tolist()
                        travel_limited = True
            entry_recovery = None
            try:
                if self.robot.motion_validation == "command_limits":
                    entry_recovery = self.geometry.entry_recovery(measured)
                if entry_recovery is not None:
                    if any(data["moving"] is not False for data in before.values()):
                        raise DeviceError(ErrorCode.LIMIT_VIOLATION, "Joint entry requires stationary feedback", stage="preflight")
                    path = [{n: np.asarray(q, dtype=float) for n, q in measured.items()}, entry_recovery]
                    # Entry is a separate acknowledged prefix. Keep the grippers
                    # unchanged and require a new observation before task motion.
                    for n, target in targets.items():
                        c = self.installation.calibration.arms[n]
                        target["gripper_opening"] = float(np.clip((before[n]["agx_gripper"]["width_m"] - c.gripper_closed_m) /
                            (c.gripper_open_m - c.gripper_closed_m), 0, 1))
                    minimum_duration = .2
                else:
                    path = self.geometry.plan(measured, targets, segment_m=params["segment_m"],
                                              max_segments=params["max_segments"], checkpoint=checkpoint)
            except DeviceError as error:
                if error.code == ErrorCode.LIMIT_VIOLATION and error.execution_outcome == "none":
                    error.stage = "preflight"
                raise
            checkpoint()
            period = 1 / self.robot.control_frequency_hz
            gripper_start = {n: (before[n]["agx_gripper"]["width_m"] - self.installation.calibration.arms[n].gripper_closed_m) /
                            (self.installation.calibration.arms[n].gripper_open_m - self.installation.calibration.arms[n].gripper_closed_m)
                            for n in before}
            gripper_duration = max(abs(targets[n]["gripper_opening"] - gripper_start[n]) /
                                   self.installation.calibration.arms[n].motion_limits.gripper_speed_fraction_s for n in before)
            minimum_duration = max(minimum_duration, gripper_duration)
            # Each candidate curve derives its geometric limits independently.
            # Only caller/gripper duration requirements survive prefix selection.
            trajectory = JointTrajectory(path, self.geometry, minimum_duration)
            available = deadline - time.monotonic() - self.robot.stop_timeout_s - self.tracking_limits.settle_timeout_s
            truncated = False
            while trajectory.duration > available:
                checkpoint()
                if len(path) <= 2:
                    raise DeviceError(ErrorCode.LIMIT_VIOLATION,
                        "No checked prefix fits the action duration before dispatch", stage="preflight")
                path = path[:max(2, (len(path) + 1) // 2)]
                trajectory = JointTrajectory(path, self.geometry, minimum_duration)
                truncated = True
            if entry_recovery is None:
                self.geometry.check_trajectory(trajectory, checkpoint=checkpoint)
            checkpoint()
            if trajectory.duration > deadline - time.monotonic() - self.robot.stop_timeout_s - self.tracking_limits.settle_timeout_s:
                raise DeviceError(ErrorCode.LIMIT_VIOLATION,
                    "Trajectory preflight exhausted the execution budget", stage="preflight")
            with self._dispatch_lock:
                checkpoint()
                # Planning can be slow. Do not execute a path from an obsolete start.
                current = self._preflight_feedback(measured, checkpoint)
                if any(np.max(np.abs(np.array(current[n]["joints_rad"]) - measured[n])) >
                       self.tracking_limits.preflight_joint_rad for n in measured):
                    raise DeviceError(ErrorCode.OBSERVATION_STALE, "Measured start changed during preflight; observe and plan a new path", stage="preflight")
                # Every next connection is checked again from measured joints.
                # Admission tolerance is not a continuous collision guarantee.
                self.geometry.check({n: current[n]["joints_rad"] for n in measured}, measured=True)
                submitted = False
                elapsed = 0.
                completed_ticks = 0
                residual_prefix = False
                duration_prefix = False
                prefix_details = None
                last_target = path[0]
                command_trace = []
                path_started_at = time.monotonic()
                next_command_at = path_started_at
                guards = {n: PoseTrackingGuard(self.tracking_limits,
                    int(data["timing"]["received_monotonic_ns"]) / 1e9, self._tracking_pose(n, data))
                    for n, data in current.items()}
                actual = current
                hold_targets = {name: {"hold_joints": path[-1][name]} for name in targets
                                if "hold_joints" in targets[name]}
                monitors = self._hold_monitors(hold_targets, current)
                arrivals = {}
                last_openings = dict(gripper_start)
                try:
                    # Near-bound entry admits only the verified legal endpoint.
                    ticks = 1 if entry_recovery is not None else max(1, math.ceil(trajectory.duration / period))
                    tick_s = trajectory.duration / ticks
                    for tick in range(1, ticks + 1):
                        def motion_checkpoint():
                            if cancel.is_set() or self._blocked.is_set() or time.monotonic() >= deadline:
                                raise InterruptedError("Path interrupted before command dispatch")

                        end = min(max(path_started_at + tick * tick_s, next_command_at),
                                  deadline - self.robot.stop_timeout_s)
                        while time.monotonic() < end:
                            if cancel.wait(min(.005, max(0., end - time.monotonic()))):
                                raise InterruptedError("Path interrupted")
                            motion_checkpoint()
                            actual = {n: self._require_feedback(n) for n in measured}
                            self.geometry.check({n: data["joints_rad"] for n, data in actual.items()}, measured=True)
                            self._update_holds(monitors, actual)
                            self._check_pose_tracking(guards, actual, segment=tick)
                        actual = {n: self._require_feedback(n) for n in measured}
                        self.geometry.check({n: data["joints_rad"] for n, data in actual.items()}, measured=True)
                        self._update_holds(monitors, actual)
                        self._check_pose_tracking(guards, actual, segment=tick)
                        measured_tick = {n: np.asarray(data["joints_rad"]) for n, data in actual.items()}
                        command_row = trajectory.at(tick / ticks)
                        if not self.geometry.check_measured_segment(measured_tick, command_row, checkpoint=motion_checkpoint):
                            prefix_details = {"tick": tick,
                                "connections": self.geometry.command_step_evidence(measured_tick, command_row)}
                            if not submitted:
                                raise DeviceError(ErrorCode.LIMIT_VIOLATION,
                                    "Measured start exceeds next command step", stage="preflight", details=prefix_details)
                            residual_prefix = True
                            break
                        if time.monotonic() + self.robot.stop_timeout_s >= deadline:
                            if not submitted:
                                raise DeviceError(ErrorCode.LIMIT_VIOLATION,
                                    "Remaining action time cannot cover stopping", stage="preflight")
                            duration_prefix = True
                            break
                        next_command_at = time.monotonic() + period
                        for name, q in command_row.items():
                            motion_checkpoint()
                            c = self.installation.calibration.arms[name]
                            gripper_fraction = min(1., tick * tick_s / max(gripper_duration, 1e-9))
                            command = gripper_start[name] + gripper_fraction * (targets[name]["gripper_opening"] - gripper_start[name])
                            opening = c.gripper_closed_m + command * (c.gripper_open_m - c.gripper_closed_m)
                            record = {"arm": name, "tick": tick, "ticks": ticks,
                                "sent_realtime_ns": str(time.time_ns()), "target_joints_rad": q.tolist(),
                                "opening_m": opening, "measured_start_rad": measured_tick[name].tolist(),
                                "send_outcome": "unknown"}
                            command_trace.append(record)
                            submitted = True
                            if name not in monitors:
                                self._hold_references.pop(name, None)
                            self._writers[name].move(q, opening, c.gripper_effort)
                            record["send_outcome"] = "transport_returned"
                            last_openings[name] = command
                            guards[name].sent(time.monotonic(), self._tracking_target(name, q))
                        final_sent_at = time.monotonic()
                        last_target = command_row
                        elapsed = tick * tick_s
                        completed_ticks = tick
                        progress(tick / ticks)
                    windows = {n: ArrivalWindow(duration_s=self.tracking_limits.settle_duration_s,
                                               max_gap_s=self.tracking_limits.settle_max_gap_s) for n in measured}
                    arrival_deadline = min(deadline - self.robot.stop_timeout_s,
                                           time.monotonic() + self.tracking_limits.settle_timeout_s)
                    while time.monotonic() < arrival_deadline:
                        actual = {n: self._require_feedback(n) for n in measured}
                        self.geometry.check({n: data["joints_rad"] for n, data in actual.items()}, measured=True)
                        self._update_holds(monitors, actual)
                        self._check_pose_tracking(guards, actual, segment=completed_ticks)
                        arrivals = {}
                        for n, data in actual.items():
                            stamp = int(data["timing"]["received_monotonic_ns"]) / 1e9
                            require_open = targets[n]["gripper_opening"] == 1. and last_openings[n] == 1.
                            within, errors = self._arrival_error(n, data, last_target[n], held="hold_joints" in targets[n],
                                require_open=require_open)
                            errors["gripper_arrival_required"] = require_open
                            errors["gripper_opening_deferred"] = targets[n]["gripper_opening"] == 1. and not require_open
                            if entry_recovery is not None:
                                c = self.installation.calibration.arms[n]
                                within = within and bool(np.all(np.asarray(data["joints_rad"]) >= c.joint_lower)
                                                         and np.all(np.asarray(data["joints_rad"]) <= c.joint_upper))
                            settled = windows[n].update(stamp, stamp >= final_sent_at and
                                data["moving"] is False and within)
                            arrivals[n] = {"verified": settled, **errors,
                                "feedback_samples": windows[n].samples,
                                "controller_target_reached": data.get("controller_target_reached")}
                        if all(row["verified"] for row in arrivals.values()):
                            for name, data in actual.items():
                                if name not in monitors:
                                    self._hold_references[name] = {
                                        "joints_rad": list(data["joints_rad"]),
                                        "grasp_pose": (self._measured_flange(name, data) @ np.asarray(
                                            self.installation.calibration.arms[name].T_flange_from_grasp)).tolist(),
                                        "basis": "last_verified_active_arrival"}
                            return {"requested_targets_reached": not (truncated or duration_prefix or travel_limited or residual_prefix or entry_recovery is not None), "robot_id": self.robot.robot_id,
                                    "prefix_reason": "joint_entry_recovery" if entry_recovery is not None else "measured_start_step" if residual_prefix else
                                        "action_duration" if truncated or duration_prefix else "travel_budget" if travel_limited else None,
                                    "prefix_details": prefix_details,
                                    "path_reference": "registered_control_site",
                                    "motion_validation": self.robot.motion_validation,
                                    "collision_checked": self.robot.motion_validation == "collision",
                                    "planned_segments": len(trajectory.knots) - 1, "command_ticks": completed_ticks, "planned_duration_s": elapsed,
                                    "arrival": arrivals,
                                    "hold_reports": self._hold_reports(monitors, actual, arrivals),
                                    "command_trace": command_trace,
                                    "tracking_limits": self.tracking_limits.to_dict(),
                                    "completion_basis": "consecutive_fresh_target_feedback_after_final_command"}
                        if cancel.wait(.01) or self._blocked.is_set():
                            raise InterruptedError("Interrupted during final arrival")
                    raise DeviceError(ErrorCode.TRACKING_ERROR, "Measured arrival was not verified",
                        stage="tracking", details={"reason": "arrival_timeout", "arrival": arrivals})
                except Exception as error:
                    if not submitted and isinstance(error, DeviceError):
                        raise
                    sends_known = all(item["send_outcome"] == "transport_returned" for item in command_trace)
                    recoverable = submitted and isinstance(error, DeviceError) and error.code == ErrorCode.TRACKING_ERROR and sends_known
                    stop = self._stop_all(recoverable=recoverable) if submitted else {"status": "not_dispatched", "physical_stop_guaranteed": False}
                    holds_sent = set(stop.get("devices", {})) == set(self._writers) and all(
                        item.get("hold_command_sent") is True for item in stop["devices"].values())
                    if (submitted and isinstance(error, DeviceError) and error.code == ErrorCode.TRACKING_ERROR
                            and holds_sent and sends_known):
                        raise DeviceError(error.code, str(error), stage="tracking", execution_outcome="partial",
                            details={**(error.details or {}), "hold_command_sent": True,
                                     "stop": stop, "command_trace": command_trace,
                                     "hold_reports": self._hold_reports(monitors, actual, {})}) from error
                    raise DeviceError(ErrorCode.OUTCOME_UNKNOWN if submitted else ErrorCode.PRECONDITION_FAILED,
                        str(error), execution_outcome="unknown" if submitted else "none",
                        details={"stop": stop, "cause": getattr(error, "details", {}), "command_trace": command_trace,
                                 "hold_reports": self._hold_reports(monitors, actual, {})},
                        stage="execution" if submitted else "preflight") from error

    def _resume_control(self, device_ids: list[str], params: dict, cancel: threading.Event) -> dict:
        if not self.execute or set(device_ids) != set(self._by_device):
            raise DeviceError(ErrorCode.PRECONDITION_FAILED, "Resume must name every enabled arm", stage="preflight")
        with self._planning_lock, self._dispatch_lock, self._stop_lock:
            stop = self._stop_receipt or {}
            if not self._resume_stop_id or params.get("stop_id") != self._resume_stop_id:
                raise DeviceError(ErrorCode.PRECONDITION_FAILED, "Tracking stop identity is unavailable or changed", stage="preflight")
            try:
                for name, writer in self._writers.items():
                    if cancel.is_set():
                        raise RuntimeError("Control recovery cancelled")
                    data = self._require_feedback(name)
                    stamp = int(data["timing"]["received_monotonic_ns"])
                    if data["moving"] is not False or stamp <= int(stop["held_at_monotonic_ns"]):
                        raise RuntimeError("Fresh stationary feedback after the group hold is required")
                    writer.resume(lambda name=name: self._require_feedback(name))
                if cancel.is_set():
                    raise RuntimeError("Control recovery cancelled")
            except Exception as error:
                for writer in self._writers.values():
                    writer.block()
                raise DeviceError(ErrorCode.PRECONDITION_FAILED, str(error), stage="preflight") from error
            self._resume_stop_id = None
            self._stop_receipt = None
            self._blocked.clear()
            return {"resumed": True, "stop_id": stop["stop_id"], "hardware_command_sent": False}

    def _stop_all(self, *, recoverable=False, permanent=False):
        self._blocked.set()
        with self._stop_lock:
            self._blocked.set()
            if permanent:
                self._resume_stop_id = None
                for writer in self._writers.values():
                    writer.block()
            if self._stop_receipt is not None:
                return self._stop_receipt
            def stop(item):
                name, writer = item
                try:
                    return name, writer.stop(self._readers[name].read)
                except Exception as error:
                    return name, {"status": "unknown", "error": str(error)}
            # This lock is independent of dispatch and stops only this Body's arms.
            with ThreadPoolExecutor(max_workers=max(1, len(self._writers))) as pool:
                evidence = dict(pool.map(stop, self._writers.items()))
            self._stop_receipt = {"robot_id": self.robot.robot_id, "devices": evidence,
                "stop_id": uuid.uuid4().hex, "held_at_monotonic_ns": str(time.monotonic_ns()),
                "status": "requested" if evidence else "not_required", "physical_stop_guaranteed": False}
            if recoverable and evidence and all(item.get("hold_command_sent") is True for item in evidence.values()):
                self._resume_stop_id = self._stop_receipt["stop_id"]
            return self._stop_receipt

    def estop(self, device_id):
        if device_id not in self._by_device:
            raise DeviceError(ErrorCode.UNKNOWN_DEVICE, "Unknown arm")
        return self._stop_all(permanent=True)

    def stream_hold(self, device_id):
        if device_id not in self._by_device:
            raise DeviceError(ErrorCode.UNKNOWN_DEVICE, "Unknown arm")
        # Body admission is quiesced before this hook. An idle position controller
        # already holds its completed target; emergency stop is not normal release.
        if not self.execute:
            return {"status": "not_required", "physical_stop_guaranteed": False}
        if self._dispatch_lock.acquire(blocking=False):
            try:
                if not self._blocked.is_set() and all(
                        self._require_feedback(name)["moving"] is False for name in self.robot.arms):
                    return {"status": "retained_position_hold", "physical_stop_guaranteed": False}
            except Exception:
                pass  # Unknown feedback must use the stop path.
            finally:
                self._dispatch_lock.release()
        return self._stop_all()

    def write(self, *args, **kwargs):
        raise DeviceError(ErrorCode.NOT_SUPPORTED, "Only preflighted group paths are supported")

    stream_apply = write

    def close(self):
        if self.execute and self._writers:
            self.stream_hold(next(iter(self._by_device)))
        self._cleanup.close()
