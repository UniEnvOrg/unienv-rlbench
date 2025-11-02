from typing import Optional, Tuple, Dict, Any, Callable, List, Type, Literal, Union
import numpy as np

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode
from rlbench.backend.task import Task
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.environment import Environment as RLBenchEnvironment
from rlbench.task_environment import TaskEnvironment as RLBenchTaskEnvironment

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import BoxSpace, DictSpace, TextSpace
from unienv_interface.space.space_utils import construct_utils as scu
from unienv_interface.env_base.env import Env

__all__ = ['FromRLBenchEnv']

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.vector_norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    return q / norm

class FromRLBenchEnv(Env[NumpyArrayType, Dict[str, Any], Dict[str, Any], NumpyArrayType, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    metadata = {
        "render_modes": ["human", "rgb_array"]
    }

    @staticmethod
    def transform_obs(
        rlbench_obs: Observation
    ) -> Dict[str, Any]:
        out_obs = {}
        for state_name in rlbench_obs.__dir__():
            if state_name.startswith('_'):
                continue

            state_data = getattr(rlbench_obs, state_name)
            if state_data is None:
                continue

            if isinstance(state_data, np.ndarray):
                new_state_data = np.float32(state_data)
            elif isinstance(state_data, (float, int)):
                new_state_data = np.array([state_data], dtype=np.float32)
            elif isinstance(state_data, dict):
                new_state_data = {}
                for k, v in state_data.items():
                    if isinstance(v, np.ndarray):
                        new_state_data[k] = np.float32(v)
                    elif isinstance(v, (float, int)):
                        new_state_data[k] = np.array([v], dtype=np.float32)
                    elif isinstance(v, Callable) or isinstance(v, List):
                        continue
                    else:
                        raise ValueError(f"Unsupported data type in observation {state_name}/{k}: {type(v)}")
            elif isinstance(state_data, Callable):
                continue
            else:
                raise ValueError(f"Unsupported data type in observation {state_name}: {type(state_data)}")
            out_obs[state_name] = new_state_data
        return out_obs
    
    @staticmethod
    def transform_action(
        action_mode: ActionMode,
        action: NumpyArrayType
    ) -> NumpyArrayType:
        if isinstance(action_mode, MoveArmThenGripper):
            if isinstance(action_mode.arm_action_mode, (EndEffectorPoseViaIK, EndEffectorPoseViaPlanning)):
                quat_input = action[..., 3:7]
                action = np.concatenate([
                    action[..., :3],
                    normalize_quaternion(quat_input),
                    action[..., 7:],
                ], axis=-1)
        return action

    def create(
        task_class: Type[Task],
        obs_config : Optional[Union[ObservationConfig, Literal['state', 'vision']]] = 'state',
        action_mode : Optional[ActionMode] = None,
        render_mode:Literal['human', 'rgb_array'] = 'rgb_array',
    ) -> 'FromRLBenchEnv':
        if not isinstance(obs_config, ObservationConfig):
            obs_conf = ObservationConfig()
            if obs_config == 'state':
                obs_conf.set_all_high_dim(False)
                obs_conf.set_all_low_dim(True)
            elif obs_config == 'vision':
                obs_conf.set_all(True)
            else:
                raise ValueError(f"Unrecognized obs_config: {obs_config}")
            obs_config = obs_conf
        rlbench_env = RLBenchEnvironment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True,
        )
        rlbench_env.launch()
        rlbench_task_env = rlbench_env.get_task(task_class)
        return __class__(
            rlbench_env,
            rlbench_task_env,
            render_mode=render_mode,
        )

    def __init__(
        self,
        runtime_env:RLBenchEnvironment,
        task_env: RLBenchTaskEnvironment,
        render_mode: Literal['human', 'rgb_array'] = 'rgb',
    ):
        self.runtime_env = runtime_env
        self.task_env = task_env

        self.backend = NumpyComputeBackend
        self.device = None

        assert render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        # Initialize rendering
        if render_mode is not None:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([640, 360])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)            
        
        # Initialize spaces
        try:
            action_bounds = task_env._action_mode.action_bounds()
        except NotImplementedError:
            action_bounds = (-np.inf, np.inf)

        self.action_space = BoxSpace(
            NumpyComputeBackend,
            low=action_bounds[0],
            high=action_bounds[1],
            dtype=np.float32,
            shape=(self.runtime_env.action_shape),
        )
        self.context_space = None
        # We need to use a sample observation to construct the observation space
        _, sample_obs = self.task_env.reset()
        sample_obs = self.transform_obs(sample_obs)
        obs_space : DictSpace = scu.construct_space_from_data(
            sample_obs,
            NumpyComputeBackend,
        )
        for key, space in obs_space.spaces.items():
            if "rgb" in key:
                obs_space[key] = BoxSpace(
                    NumpyComputeBackend,
                    low=0,
                    high=255,
                    shape=space.shape,
                    dtype=np.uint8,
                )
        self.observation_space = obs_space

        # Initialize RNG
        self.rng = NumpyComputeBackend.random.random_number_generator()

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode

    @property
    def render_fps(self) -> Optional[int]:
        return int(round(1 / self.runtime_env._pyrep.get_simulation_timestep()))
    
    @property
    def batch_size(self) -> Optional[int]:
        return None
    
    def step(self, action):
        action = self.transform_action(self.task_env._action_mode, action)
        obs, reward, terminated = self.task_env.step(action)
        return self.transform_obs(obs), reward, terminated, False, {}

    def reset(self, *, mask = None, seed = None, target_demo : Optional[Demo] = None) -> Tuple[None, Dict[str, Any], Dict]:
        assert mask is None, "Masking not supported for a non-batched environment."
        if seed is not None:
            np.random.seed(seed=seed)

        if target_demo is not None:
            language_descriptions, obs = self.task_env.reset_to_demo(target_demo)
        else:
            language_descriptions, obs = self.task_env.reset()
        obs = self.transform_obs(obs)
        return None, obs, {}
    
    def get_live_demos(
        self,
        amount : int,
        max_attempts : int = 10,
        callable_each_step: Optional[Callable[[Observation], None]] = None,
    ) -> List[Demo]:
        return self.task_env.get_demos(
            amount,
            live_demos=True,
            max_attempts=max_attempts,
            callable_each_step=callable_each_step,
        )

    def render(self):
        if self.render_mode == 'rgb_array':
            frame = self.gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame
    
    def close(self):
        self.runtime_env.shutdown()
