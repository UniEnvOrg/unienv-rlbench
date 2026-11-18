# Changelog

## 0.0.1a1 (unreleased)

- `FromRLBenchEnv`: RLBench tasks as UniEnv environments with space metadata; the episode's language
  instruction as context; seeded resets that reproduce the episode; `variation` (fixed or `"random"`);
  `max_episode_steps` truncation; `info["success"]`; demonstrations (`get_live_demos`,
  `reset(target_demo=...)`); `render()`.
- Observations no longer gain `misc/joint_position_action` after live demonstrations were recorded.
- `import unienv_rlbench` and `unienv_rlbench.space_metadata` work without RLBench / PyRep; using
  `FromRLBenchEnv` without them raises an `ImportError` pointing to the installation steps.
