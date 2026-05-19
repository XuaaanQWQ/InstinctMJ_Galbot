from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from instinct_mj.envs.scene import InstinctScene
from instinct_mj.managers import MultiRewardCfg, MultiRewardManager
from instinct_mj.monitors import MonitorManager


def _log_rank_stage(stage: str) -> None:
    rank = os.environ.get("RANK", "0")
    world_size = os.environ.get("WORLD_SIZE", "1")
    message = f"[INFO rank {rank}/{world_size}] env: {stage}"
    print(message, flush=True)
    rank_log_dir = os.environ.get("INSTINCT_RANK_LOG_DIR")
    if rank_log_dir is None:
        return
    path = Path(rank_log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"rank_{rank}.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} env: {stage}\n")


class InstinctRlEnv(ManagerBasedRlEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def __init__(
        self,
        cfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        del kwargs  # Unused.
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self._instinct_body_lin_acc_cache: dict[str, dict[str, object]] = {}
        self.extras = {}
        self.obs_buf = {}
        # Initialize the manual-reset state here because InstinctRlEnv
        # customizes scene construction instead of calling ManagerBasedRlEnv.__init__.
        self._manual_reset_pending = torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=device)

        # Use InstinctScene so terrain cfg.class_type is honored (e.g. hacked_generator importer).
        _log_rank_stage("scene build start")
        self.scene = InstinctScene(self.cfg.scene, device=device)
        _log_rank_stage("scene build done")
        _log_rank_stage("simulation build start")
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=device,
        )
        _log_rank_stage("simulation build done")

        _log_rank_stage("scene initialize start")
        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )
        _log_rank_stage("scene initialize done")
        if self.scene.sensor_context is not None:
            _log_rank_stage("sensor context set start")
            self.sim.set_sensor_context(self.scene.sensor_context)
            _log_rank_stage("sensor context set done")

        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=device, dtype=torch.long)
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene)
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt

        _log_rank_stage("load managers start")
        self.load_managers()
        _log_rank_stage("load managers done")
        _log_rank_stage("setup manager visualizers start")
        self.setup_manager_visualizers()
        _log_rank_stage("setup manager visualizers done")

    def load_managers(self) -> None:
        # Route Instinct tasks through MultiRewardManager so reward logging matches
        # InstinctLab conventions:
        #   Episode_Reward/rewards_<term>/{max_episode_len_s,sum,timestep}
        reward_group_cfg = self._as_multi_reward_cfg(self.cfg.rewards)
        if reward_group_cfg is not None:
            self.cfg.rewards = {}

        super().load_managers()

        # Replace parent reward manager with MultiRewardManager when requested.
        if reward_group_cfg is not None:
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt)
            print_info(f"[INFO] {self.reward_manager}")

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)

    @staticmethod
    def _as_multi_reward_cfg(rewards_cfg):
        """Convert reward config into a multi-reward group config when possible.

        - Keep existing MultiRewardCfg as-is.
        - For flat dict[str, RewardTermCfg], wrap into {"rewards": ...}.
        - For grouped dicts (dict[str, dict[str, RewardTermCfg]]), keep as-is.
        """
        if isinstance(rewards_cfg, MultiRewardCfg):
            return rewards_cfg
        if isinstance(rewards_cfg, dict):
            first_non_none = next(
                (value for value in rewards_cfg.values() if value is not None),
                None,
            )
            if first_non_none is None or isinstance(first_non_none, RewardTermCfg):
                return {"rewards": rewards_cfg}
            return rewards_cfg
        return None

    def setup_manager_visualizers(self) -> None:
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = self.monitor_manager

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        extras.setdefault("step", {})
        extras["step"].update(monitor_infos)
        return obs, reward, terminated, truncated, extras

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[object, dict]:
        del options  # Unused.
        _log_rank_stage("reset start")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if seed is not None:
            _log_rank_stage("reset seed start")
            self.seed(seed)
            _log_rank_stage("reset seed done")

        _log_rank_stage("reset _reset_idx start")
        self._reset_idx(env_ids)
        _log_rank_stage("reset _reset_idx done")

        _log_rank_stage("reset scene.write_data_to_sim start")
        self.scene.write_data_to_sim()
        _log_rank_stage("reset scene.write_data_to_sim done")

        _log_rank_stage("reset sim.forward start")
        self.sim.forward()
        _log_rank_stage("reset sim.forward done")

        _log_rank_stage("reset command_manager.compute start")
        self.command_manager.compute(dt=0.0)
        _log_rank_stage("reset command_manager.compute done")

        _log_rank_stage("reset sim.sense start")
        self._sense_with_rank_logging()
        _log_rank_stage("reset sim.sense done")

        _log_rank_stage("reset observation_manager.compute start")
        self.obs_buf = self.observation_manager.compute(update_history=True)
        _log_rank_stage("reset observation_manager.compute done")

        _log_rank_stage("reset recorder_manager.record_post_reset start")
        self.recorder_manager.record_post_reset(env_ids)
        _log_rank_stage("reset recorder_manager.record_post_reset done")
        _log_rank_stage("reset done")
        return self.obs_buf, self.extras

    def _sense_with_rank_logging(self) -> None:
        if self.sim._sensor_context is None:
            _log_rank_stage("sense skipped no sensor context")
            return

        ctx = self.sim._sensor_context
        _log_rank_stage("sense prepare start")
        ctx.prepare()
        _log_rank_stage("sense prepare done")

        _log_rank_stage("sense kernel start")
        with wp.ScopedDevice(self.sim.wp_device):
            # Avoid sensing CUDA graph hangs seen in multi-process distributed startup.
            self.sim._sense_kernel()
        _log_rank_stage("sense kernel done")

        _log_rank_stage("sense finalize start")
        ctx.finalize()
        _log_rank_stage("sense finalize done")

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        terrain = self.scene.terrain
        if terrain is not None:
            terrain.debug_vis(visualizer)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if isinstance(env_ids, Sequence):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)

        _log_rank_stage("_reset_idx monitor_manager.reset start")
        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
        _log_rank_stage("_reset_idx monitor_manager.reset done")

        _log_rank_stage("_reset_idx curriculum_manager.compute start")
        self.curriculum_manager.compute(env_ids=env_ids)
        _log_rank_stage("_reset_idx curriculum_manager.compute done")

        _log_rank_stage("_reset_idx sim.reset start")
        self.sim.reset(env_ids)
        _log_rank_stage("_reset_idx sim.reset done")

        _log_rank_stage("_reset_idx scene.reset start")
        self.scene.reset(env_ids)
        _log_rank_stage("_reset_idx scene.reset done")

        if "reset" in self.event_manager.available_modes:
            _log_rank_stage("_reset_idx event_manager.apply reset start")
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )
            _log_rank_stage("_reset_idx event_manager.apply reset done")

        self.extras["log"] = dict()

        _log_rank_stage("_reset_idx observation_manager.reset start")
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx observation_manager.reset done")

        _log_rank_stage("_reset_idx action_manager.reset start")
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx action_manager.reset done")

        _log_rank_stage("_reset_idx reward_manager.reset start")
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx reward_manager.reset done")

        _log_rank_stage("_reset_idx metrics_manager.reset start")
        info = self.metrics_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx metrics_manager.reset done")

        _log_rank_stage("_reset_idx curriculum_manager.reset start")
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx curriculum_manager.reset done")

        _log_rank_stage("_reset_idx command_manager.reset start")
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx command_manager.reset done")

        _log_rank_stage("_reset_idx event_manager.reset start")
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx event_manager.reset done")

        _log_rank_stage("_reset_idx termination_manager.reset start")
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        _log_rank_stage("_reset_idx termination_manager.reset done")

        _log_rank_stage("_reset_idx episode buffers reset start")
        self.episode_length_buf[env_ids] = 0
        self._manual_reset_pending[env_ids] = False
        _log_rank_stage("_reset_idx episode buffers reset done")

        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)

    """
  Properties.
  """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)
