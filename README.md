# unienv-rlbench

[RLBench](https://github.com/stepjam/RLBench) tasks as [UniEnv](https://github.com/UniEnvOrg/UniEnv)
environments, with UniEnv space metadata.

```python
from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.tasks import ReachTarget
from unienv_rlbench import FromRLBenchEnv

env = FromRLBenchEnv.create(ReachTarget, obs_config="vision", action_mode=JointPositionActionMode(),
                            max_episode_steps=200)
instruction, obs, info = env.reset(seed=0)        # context: the episode's language instruction
obs, reward, terminated, truncated, info = env.step(env.sample_action())
env.close()
```

- `obs_config`: `"state"` (low-dimensional), `"vision"` (all cameras: RGB, depth, point clouds, masks) or an
  RLBench `ObservationConfig`. The observation space is inferred from a first observation and described
  with space metadata (`unienv_rlbench.space_metadata`): image formats, depth in meters or normalized
  between the clipping planes, world-frame point clouds, joint states, `gripper_pose` (quaternion xyzw),
  camera intrinsics / extrinsics. The action space records the arm / gripper action mode.
- The context is the episode's language instruction; `info["descriptions"]` has all of RLBench's
  descriptions and `info["variation"]` the task variation.
- `variation`: a fixed task variation (default 0) or `"random"` to sample one at each reset.
- `reset(seed=s)` reproduces the episode (object placement, variation, instruction); the robot's
  proprioception can still differ slightly between resets (CoppeliaSim keeps some dynamics state).
- `terminated` is task success (`info["success"]`); `truncated` after `max_episode_steps`.
- `reset(target_demo=demo)` starts from a demonstration; `get_live_demos(n)` records demonstrations.
- `render()` returns an RGB frame from a 640x360 camera (`render_mode="rgb_array"`).

`unienv_rlbench.space_metadata` (and `import unienv_rlbench`) work without RLBench.

## Installation

RLBench, PyRep and CoppeliaSim are not on PyPI, so they are installed separately (Python 3.10 works):

```bash
pip install unienv_rlbench

# CoppeliaSim 4.1.0 (Ubuntu 20.04 build; also runs on Ubuntu 22.04 / 24.04)
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
export COPPELIASIM_ROOT=$PWD/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# PyRep (its pinned cffi 1.14.2 has no wheels for Python >= 3.10: install a current cffi first)
pip install cffi numpy
git clone https://github.com/stepjam/PyRep && pip install --no-build-isolation ./PyRep

# RLBench (it imports gymnasium without declaring it)
pip install git+https://github.com/stepjam/RLBench.git gymnasium
```

Headless machines need an X server for the cameras, e.g. `xvfb-run -a python script.py`.

## Tests

`pytest tests` runs the metadata tests anywhere; with RLBench installed the simulator tests run too
(`xvfb-run -a pytest tests` on a headless machine; the vision configurations render on the CPU there and
take a few minutes each).

## License

Apache-2.0. RLBench, PyRep and CoppeliaSim are installed separately and keep their own licenses
(CoppeliaSim Edu is free for educational use).
