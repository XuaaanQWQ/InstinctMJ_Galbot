from __future__ import annotations

__all__ = [
    "EdgeCylinderCfg",
    "FeatureEdgeCylinderCfg",
    "GreedyconcatEdgeCylinderCfg",
    "PluckerEdgeCylinderCfg",
    "RansacEdgeCylinderCfg",
    "RayEdgeCylinderCfg",
    "VirtualObstacleBase",
    "VirtualObstacleCfg",
]


def __getattr__(name: str):
    if name in {
        "EdgeCylinderCfg",
        "FeatureEdgeCylinderCfg",
        "GreedyconcatEdgeCylinderCfg",
        "PluckerEdgeCylinderCfg",
        "RansacEdgeCylinderCfg",
        "RayEdgeCylinderCfg",
    }:
        from . import edge_cylinder_cfg

        return getattr(edge_cylinder_cfg, name)
    if name in {"VirtualObstacleBase", "VirtualObstacleCfg"}:
        from . import virtual_obstacle_base

        return getattr(virtual_obstacle_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
