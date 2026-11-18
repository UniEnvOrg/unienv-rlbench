import traceback
import numpy as np
import pytest
from rlbench.tasks import *
from rlbench.action_modes.action_mode import *
from rlbench.action_modes.arm_action_modes import *
from rlbench.action_modes.gripper_action_modes import *
from unienv_rlbench import FromRLBenchEnv

@pytest.mark.parametrize(
    "task_class", MT15_V1['train']
)
@pytest.mark.parametrize(
    "obs_config", ['state', 'vision']
)
@pytest.mark.parametrize(
    "action_mode", [
        JointPositionActionMode()
    ]
)
def test_rlbench(
    task_class,
    obs_config,
    action_mode
):
    try:
        env : FromRLBenchEnv = FromRLBenchEnv.create(
            task_class=task_class,
            obs_config=obs_config,
            action_mode=action_mode,
            render_mode='rgb_array'
        )
    except Exception as e:
        print("Failed to create environment.")
        print(e)
        print(traceback.format_exc())
        raise e
    
    try:
        ctx, obs, info = env.reset()
    except Exception as e:
        print("Failed to reset environment.")
        print(e)
        print(traceback.format_exc())
        raise e

    assert (ctx is None and env.context_space is None) or env.context_space.contains(ctx)
    assert env.observation_space.contains(obs)

    done = False
    step_count = 0
    while not done and step_count < 100:
        action = env.sample_action()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print("Error occurred while taking a step.")
            print(e)
            print(traceback.format_exc())
            raise e
        assert env.observation_space.contains(obs)
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
    env.close()

def test_observations_match_the_space_after_live_demos():
    # RLBench keeps adding misc/joint_position_action to observations once live demos
    # were collected; the env's observations must still match its observation space.
    env = FromRLBenchEnv.create(
        task_class=ReachTarget, obs_config='state', action_mode=JointPositionActionMode(),
    )
    try:
        demo = env.get_live_demos(1)[0]
        assert len(demo) > 0
        _, obs, _ = env.reset(target_demo=demo)
        assert env.observation_space.contains(obs)
        _, obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(env.sample_action())
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_render_mode_defaults_to_rgb_array():
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    runtime = Environment(action_mode=JointPositionActionMode(), obs_config=obs_config, headless=True)
    runtime.launch()
    env = FromRLBenchEnv(runtime, runtime.get_task(ReachTarget))
    try:
        assert env.render_mode == 'rgb_array'
        frame = env.render()
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
    finally:
        env.close()
