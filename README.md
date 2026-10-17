# RLBench Environment Wrapper

Under development.



## Running Headless

Try using `xvfb-run`

## Space metadata

`FromRLBenchEnv` describes its spaces with unienv space metadata (requires
`unienv>=0.0.1b16`). The observation space is inferred from a sample
observation and then annotated by `unienv_rlbench.space_metadata`, keyed on
RLBench's `Observation` fields: camera images / depth (meters, or normalized
between the clipping planes) / world-frame point clouds / masks, joint states,
`gripper_pose` (world frame, quaternion xyzw) and the `misc` camera
parameters. The action space records the arm / gripper action mode
(`control`, `mode`, `frame`, `rotation`, per-element `names` / `units`).
