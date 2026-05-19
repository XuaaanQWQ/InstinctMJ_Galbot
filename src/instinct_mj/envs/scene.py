from __future__ import annotations

from mjlab.scene import Scene
from mjlab.terrains import TerrainEntity


class InstinctScene(Scene):
    """Scene variant that honors terrain cfg.class_type in Instinct tasks."""

    def _add_terrain(self) -> None:
        if self._cfg.terrain is None:
            return
        terrain_cfg = self._cfg.terrain
        terrain_cfg.num_envs = self._cfg.num_envs
        terrain_cfg.env_spacing = self._cfg.env_spacing
        if terrain_cfg.class_type is not None and terrain_cfg.class_type is not TerrainEntity:
            terrain = terrain_cfg.class_type(terrain_cfg, device=self._device)
        else:
            terrain = TerrainEntity(terrain_cfg, device=self._device)
        self._terrain = terrain
        self._entities["terrain"] = terrain
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(terrain.spec, prefix="", frame=frame)
