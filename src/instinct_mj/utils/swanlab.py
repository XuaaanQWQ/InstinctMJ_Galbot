"""SwanLab logging utilities for the instinct_rl training runner."""

from __future__ import annotations

import os
from typing import Any, cast

from torch.utils.tensorboard import SummaryWriter


class SwanLabSummaryWriter(SummaryWriter):
    """TensorBoard-compatible writer that also uploads scalars to SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict[str, Any]) -> None:
        super().__init__(log_dir, flush_secs=flush_secs)
        try:
            import swanlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError("swanlab package is required. Install with: pip install swanlab") from None

        project = str(cfg.get("swanlab_project") or "InstinctMJ_Galbot")
        experiment_name = os.path.split(log_dir)[-1]
        swanlab.init(
            project=project,
            experiment_name=experiment_name,
            logdir=os.path.join(log_dir, "swanlog"),
            config={"log_dir": log_dir, "train_cfg": cfg},
        )
        self._swanlab = swanlab

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
            double_precision=double_precision,
        )
        self._swanlab.log({tag: scalar_value}, step=global_step)

    def close(self) -> None:
        super().close()
        self._swanlab.finish()


def patch_instinct_rl_swanlab_logger(train_cfg: dict[str, Any]) -> None:
    """Patch instinct_rl's module-level SummaryWriter before runner.learn()."""
    from instinct_rl.runners import on_policy_runner as runner_mod

    class _ConfiguredSwanLabSummaryWriter(SwanLabSummaryWriter):
        def __init__(self, log_dir: str, flush_secs: int = 10, **_: Any) -> None:
            super().__init__(log_dir=log_dir, flush_secs=flush_secs, cfg=train_cfg)

    runner_mod.SummaryWriter = cast(Any, _ConfiguredSwanLabSummaryWriter)
