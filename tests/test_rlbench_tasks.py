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
    env : FromRLBenchEnv = FromRLBenchEnv.create(
        task_class=task_class,
        obs_config=obs_config,
        action_mode=action_mode,
        render_mode='rgb_array'
    )
    
    ctx, obs, info = env.reset()

    assert (ctx is None and env.context_space is None) or env.context_space.contains(ctx)
    assert env.observation_space.contains(obs)

    done = False
    step_count = 0
    while not done and step_count < 100:
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
    env.close()