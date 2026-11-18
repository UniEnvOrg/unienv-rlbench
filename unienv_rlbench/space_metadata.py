"""Space metadata for RLBench observations and actions.

RLBench spaces are inferred from a sample observation at runtime, so the
metadata is attached afterwards (with ``annotate``), keyed on the fields of
``rlbench.backend.observation.Observation`` as flattened by
``FromRLBenchEnv.transform_obs``. This module only depends on unienv (not on
RLBench / PyRep), so it can be used and tested without CoppeliaSim.

Conventions (RLBench / PyRep):

- ``{camera}_rgb``: uint8 (H, W, 3) RGB; ``{camera}_depth``: (H, W) in meters
  with ``CameraConfig.depth_in_meters``, otherwise normalized to [0, 1]
  linearly between the camera's near and far clipping planes (recorded as
  ``range``); ``{camera}_point_cloud``: (H, W, 3) world-frame points in
  meters; ``{camera}_mask``: (H, W) object handles (``masks_as_one_channel``,
  the default) or (H, W, 3) RGB-coded handles.
- ``gripper_pose``: tip position (m) + quaternion, both in the world frame;
  PyRep's ``Object.get_pose`` returns ``[x, y, z, qx, qy, qz, qw]``.
- Joint positions / velocities / forces of the arm in rad / rad/s / N*m.
- ``misc``: per-camera ``*_camera_extrinsics`` (4x4 camera pose in the world
  frame), ``*_camera_intrinsics`` (3x3), ``*_camera_near`` / ``*_camera_far``
  (m), plus ``variation_index`` and, in the observations of recorded demos,
  ``joint_position_action`` (``FromRLBenchEnv`` drops it from env observations).
"""

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    depth_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)

__all__ = [
    "RLBENCH_CAMERAS",
    "rlbench_observation_metadata",
    "rlbench_action_metadata",
]

#: The cameras of an RLBench scene (``ObservationConfig`` attribute prefixes).
RLBENCH_CAMERAS = ("left_shoulder", "right_shoulder", "overhead", "wrist", "front")

_POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
_POSE_UNITS = ["m", "m", "m", "normalized", "normalized", "normalized", "normalized"]


def _scalar(value: Any) -> Optional[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return float(array[0]) if array.size == 1 and np.isfinite(array[0]) else None


def rlbench_observation_metadata(
    obs: Mapping[str, Any],
    depth_in_meters: Optional[Mapping[str, Optional[bool]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """``annotate`` mapping for an observation produced by ``FromRLBenchEnv.transform_obs``.

    Only the paths present in ``obs`` are returned. ``depth_in_meters`` maps a
    camera name to its ``CameraConfig.depth_in_meters``; for a camera missing
    from it the depth units are unknown and left out.
    """
    depth_in_meters = dict(depth_in_meters or {})
    misc = obs.get("misc", {}) if isinstance(obs.get("misc", {}), Mapping) else {}
    table: Dict[str, Dict[str, Any]] = {}

    for camera in RLBENCH_CAMERAS:
        if f"{camera}_rgb" in obs:
            table[f"{camera}_rgb"] = image_metadata(source=camera)
        if f"{camera}_depth" in obs:
            in_meters = depth_in_meters.get(camera)
            if in_meters:
                table[f"{camera}_depth"] = depth_metadata(source=camera)
            elif in_meters is None:
                table[f"{camera}_depth"] = depth_metadata(
                    units=None, source=camera,
                    label="depth (units unknown: depth_in_meters not known)",
                )
            else:
                near = _scalar(misc.get(f"{camera}_camera_near"))
                far = _scalar(misc.get(f"{camera}_camera_far"))
                table[f"{camera}_depth"] = depth_metadata(
                    units="normalized", normalized=True, source=camera,
                    range=[near, far] if near is not None and far is not None and near <= far else None,
                    label="depth normalized linearly between the near and far clipping planes",
                )
        if f"{camera}_point_cloud" in obs:
            table[f"{camera}_point_cloud"] = metadata(
                "pointcloud", units="m", frame="world", layout="HWC",
                names=["x", "y", "z"], source=camera,
            )
        if f"{camera}_mask" in obs:
            # One channel (the ObservationConfig default): object handles;
            # otherwise the raw RGB-coded handles.
            one_channel = np.ndim(obs[f"{camera}_mask"]) == 2
            table[f"{camera}_mask"] = metadata(
                "segmentation", layout="HW" if one_channel else "HWC", source=camera,
                encoding="object_handle" if one_channel else "rgb_coded_handle",
            )
        for suffix, md in (
            ("extrinsics", metadata(
                "extrinsics", frame="world", rotation="matrix", units="m", source=camera,
                label="camera pose in the world frame (4x4, camera to world)",
            )),
            ("intrinsics", metadata("intrinsics", source=camera)),
            ("near", metadata("intrinsics", units="m", source=camera, label="near clipping plane")),
            ("far", metadata("intrinsics", units="m", source=camera, label="far clipping plane")),
        ):
            if f"{camera}_camera_{suffix}" in misc:
                table[f"misc/{camera}_camera_{suffix}"] = md

    fields = {
        "joint_positions": proprio_metadata(units="rad"),
        "joint_velocities": proprio_metadata(units="rad/s"),
        "joint_forces": metadata("force", units="N*m", label="arm joint torques"),
        "gripper_open": proprio_metadata(names=["gripper_open"], label="1 open, 0 closed"),
        "gripper_pose": proprio_metadata(
            frame="world", rotation="quat_xyzw", names=_POSE_NAMES, units=_POSE_UNITS,
            label="gripper tip pose",
        ),
        "gripper_matrix": metadata(
            "pose", frame="world", rotation="matrix", units="m",
            label="gripper tip pose (4x4 homogeneous transform)",
        ),
        "gripper_joint_positions": proprio_metadata(units="m", label="gripper finger joint positions"),
        "gripper_touch_forces": metadata("force", units="N", label="gripper touch sensor forces"),
        "task_low_dim_state": metadata("privileged", label="task-specific low-dimensional state"),
        "ignore_collisions": metadata("state", label="ignore_collisions flag of the current waypoint"),
    }
    for key, md in fields.items():
        if key in obs:
            table[key] = md
    if "variation_index" in misc:
        table["misc/variation_index"] = metadata("task", label="task variation index")
    if "joint_position_action" in misc:
        table["misc/joint_position_action"] = action_metadata(
            control="joint_position", mode="absolute",
            label="joint positions requested during demo collection",
        )
    return table


def rlbench_action_metadata(
    size: int,
    arm: Optional[str],
    *,
    absolute: Optional[bool] = None,
    frame: Optional[str] = None,
    gripper: Optional[str] = None,
) -> Dict[str, Any]:
    """Metadata of an RLBench action of ``size`` elements: the arm command followed by the gripper's.

    ``arm`` is ``"eef_pose"`` (``EndEffectorPoseViaIK`` / ``EndEffectorPoseViaPlanning``:
    position + quaternion xyzw), ``"joint_position"``, ``"joint_velocity"``,
    ``"joint_torque"`` or ``None`` (unknown). ``absolute`` is the arm mode's
    ``absolute_mode``, ``frame`` the end-effector modes' ``frame``
    (``"world"`` / ``"end effector"``). ``gripper`` is ``"discrete"``
    (``Discrete``: > 0.5 opens), ``"joint_position"`` (``GripperJointPosition``,
    finger joint position in m) or ``None``. Per-element ``names`` / ``units``
    are only given when the layout is known and matches ``size``.
    """
    md: Dict[str, Any] = {}
    if arm == "eef_pose":
        md.update(control="eef_pose", rotation="quat_xyzw")
        if frame is not None:
            md["frame"] = "world" if frame == "world" else "link_local"
            if frame != "world":
                md["link"] = "end_effector"
    elif arm == "joint_position":
        md["control"] = "joint_position"
    elif arm == "joint_velocity":
        md.update(control="joint_velocity", mode="velocity")
    elif arm == "joint_torque":
        md.update(control="joint_torque", mode="absolute")
    if arm in ("eef_pose", "joint_position") and absolute is not None:
        md["mode"] = "absolute" if absolute else "delta"
    if gripper is not None:
        md["gripper"] = gripper

    gripper_units = {"discrete": "normalized", "joint_position": "m"}.get(gripper)
    if arm is not None and gripper_units is not None:
        n_arm = size - 1
        if arm == "eef_pose":
            names: Optional[Sequence[str]] = _POSE_NAMES if n_arm == 7 else None
            units: Optional[Sequence[str]] = _POSE_UNITS if n_arm == 7 else None
        else:
            names = [f"joint_{i}" for i in range(n_arm)] if n_arm > 0 else None
            unit = {"joint_position": "rad", "joint_velocity": "rad/s", "joint_torque": "N*m"}[arm]
            units = [unit] * n_arm if n_arm > 0 else None
        if names is not None and units is not None:
            md["names"] = list(names) + ["gripper"]
            md["units"] = list(units) + [gripper_units]
    return action_metadata(**md)
