"""RLBench space metadata on synthetic observations (no RLBench / CoppeliaSim needed).

``unienv_rlbench/__init__.py`` imports the PyRep-based wrapper, so the
dependency-free ``space_metadata`` module is loaded from its file directly.
The synthetic observation follows the layout ``FromRLBenchEnv.transform_obs``
produces from an ``rlbench.backend.observation.Observation``.
"""

import importlib.util
import os

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.space import BoxSpace
from unienv_interface.space.space_utils import construct_utils as scu
from unienv_interface.space.space_utils.metadata_utils import annotate, validate_metadata

_spec = importlib.util.spec_from_file_location(
    "unienv_rlbench_space_metadata",
    os.path.join(os.path.dirname(__file__), "..", "unienv_rlbench", "space_metadata.py"),
)
space_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(space_metadata)

H, W = 8, 10


def _synthetic_obs(cameras=space_metadata.RLBENCH_CAMERAS, one_channel_masks=True):
    obs = {}
    misc = {}
    for camera in cameras:
        obs[f"{camera}_rgb"] = np.zeros((H, W, 3), dtype=np.uint8)
        obs[f"{camera}_depth"] = np.zeros((H, W), dtype=np.float32)
        obs[f"{camera}_point_cloud"] = np.zeros((H, W, 3), dtype=np.float32)
        obs[f"{camera}_mask"] = (
            np.zeros((H, W), dtype=np.int64) if one_channel_masks else np.zeros((H, W, 3), dtype=np.float32)
        )
        misc[f"{camera}_camera_extrinsics"] = np.eye(4, dtype=np.float32)
        misc[f"{camera}_camera_intrinsics"] = np.eye(3, dtype=np.float32)
        misc[f"{camera}_camera_near"] = np.array([0.01], dtype=np.float32)
        misc[f"{camera}_camera_far"] = np.array([4.5], dtype=np.float32)
    misc["variation_index"] = np.array([0.0], dtype=np.float32)
    obs.update({
        "joint_velocities": np.zeros(7, dtype=np.float32),
        "joint_positions": np.zeros(7, dtype=np.float32),
        "joint_forces": np.zeros(7, dtype=np.float32),
        "gripper_open": np.array([1.0], dtype=np.float32),
        "gripper_pose": np.zeros(7, dtype=np.float32),
        "gripper_matrix": np.eye(4, dtype=np.float32),
        "gripper_joint_positions": np.zeros(2, dtype=np.float32),
        "task_low_dim_state": np.zeros(13, dtype=np.float32),
        "misc": misc,
    })
    return obs


def _space(obs):
    # Mirrors FromRLBenchEnv.transform_obsspace.
    space = scu.construct_space_from_data(obs, NumpyComputeBackend)
    for key, sub in space.spaces.items():
        if "rgb" in key:
            space[key] = BoxSpace(NumpyComputeBackend, 0, 255, np.uint8, shape=sub.shape)
    return space


def test_vision_observation_is_fully_described():
    obs = _synthetic_obs()
    depth_cfg = {camera: camera != "front" for camera in space_metadata.RLBENCH_CAMERAS}
    space = annotate(_space(obs), space_metadata.rlbench_observation_metadata(obs, depth_cfg))
    validate_metadata(space)

    # Every leaf is described.
    for key, sub in space.spaces.items():
        if key != "misc":
            assert sub.metadata.get("kind"), key
    for key, sub in space["misc"].spaces.items():
        assert sub.metadata.get("kind"), key

    rgb = space["wrist_rgb"].metadata
    assert (rgb["kind"], rgb["format"], rgb["layout"], rgb["source"]) == ("image", "rgb", "HWC", "wrist")
    meters = space["wrist_depth"].metadata
    assert (meters["kind"], meters["units"], meters["layout"]) == ("depth", "m", "HW")
    normalized = space["front_depth"].metadata
    assert normalized["units"] == "normalized" and normalized["normalized"] is True
    assert tuple(normalized["range"]) == pytest.approx((0.01, 4.5))
    pcd = space["overhead_point_cloud"].metadata
    assert (pcd["kind"], pcd["frame"], pcd["units"]) == ("pointcloud", "world", "m")
    mask = space["left_shoulder_mask"].metadata
    assert (mask["kind"], mask["layout"], mask["encoding"]) == ("segmentation", "HW", "object_handle")

    pose = space["gripper_pose"].metadata
    assert (pose["kind"], pose["frame"], pose["rotation"]) == ("proprio", "world", "quat_xyzw")
    assert tuple(pose["names"]) == ("x", "y", "z", "qx", "qy", "qz", "qw")
    assert space["joint_positions"].metadata["units"] == "rad"
    assert space["joint_velocities"].metadata["units"] == "rad/s"
    assert space["joint_forces"].metadata["units"] == "N*m"
    assert space["task_low_dim_state"].metadata["kind"] == "privileged"
    ext = space["misc"]["front_camera_extrinsics"].metadata
    assert (ext["kind"], ext["frame"], ext["source"]) == ("extrinsics", "world", "front")
    assert space["misc"]["front_camera_near"].metadata["units"] == "m"


def test_state_observation_and_unknown_depth_units():
    # 'state' observations have no cameras at all.
    state_obs = {key: value for key, value in _synthetic_obs(cameras=()).items()}
    space = annotate(_space(state_obs), space_metadata.rlbench_observation_metadata(state_obs))
    validate_metadata(space)
    assert space["gripper_pose"].metadata["rotation"] == "quat_xyzw"

    # Unknown depth_in_meters: no units; RGB-coded masks keep their channel axis.
    obs = _synthetic_obs(cameras=("front",), one_channel_masks=False)
    space = annotate(_space(obs), space_metadata.rlbench_observation_metadata(obs))
    validate_metadata(space)
    assert "units" not in space["front_depth"].metadata
    assert space["front_mask"].metadata["layout"] == "HWC"


@pytest.mark.parametrize(
    "description, expected",
    [
        (
            dict(arm="eef_pose", absolute=True, frame="world", gripper="discrete"),
            dict(control="eef_pose", mode="absolute", frame="world", rotation="quat_xyzw"),
        ),
        (
            dict(arm="eef_pose", absolute=False, frame="end effector", gripper="discrete"),
            dict(control="eef_pose", mode="delta", frame="link_local", rotation="quat_xyzw"),
        ),
        (
            dict(arm="joint_position", absolute=False, gripper="joint_position"),
            dict(control="joint_position", mode="delta"),
        ),
        (dict(arm="joint_velocity", gripper="discrete"), dict(control="joint_velocity", mode="velocity")),
        (dict(arm="joint_torque", gripper="discrete"), dict(control="joint_torque", mode="absolute")),
    ],
)
def test_action_metadata(description, expected):
    md = space_metadata.rlbench_action_metadata(8, **description)
    action_space = BoxSpace(NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(8,), metadata=md)
    validate_metadata(action_space)
    assert md["kind"] == "action"
    for key, value in expected.items():
        assert md[key] == value, key
    assert len(md["names"]) == 8 and md["names"][-1] == "gripper"
    assert len(md["units"]) == 8


def test_action_metadata_unknown_modes_stay_generic():
    md = space_metadata.rlbench_action_metadata(9, arm=None)
    assert md == {"kind": "action"}
    # A layout that does not match the size gets no per-element lists.
    md = space_metadata.rlbench_action_metadata(9, arm="eef_pose", absolute=True, gripper="discrete")
    assert "names" not in md and "units" not in md
    validate_metadata(BoxSpace(NumpyComputeBackend, -1.0, 1.0, np.float32, shape=(9,), metadata=md))
