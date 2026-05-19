from __future__ import annotations

_HEIGHT_FIELD_EXPORTS = {
    "PerlinCrossStoneTerrainCfg",
    "PerlinDiscreteObstaclesTerrainCfg",
    "PerlinGutterTerrainCfg",
    "PerlinInvertedPyramidSlopedTerrainCfg",
    "PerlinInvertedPyramidStairsTerrainCfg",
    "PerlinParapetTerrainCfg",
    "PerlinPlaneTerrainCfg",
    "PerlinPyramidSlopedTerrainCfg",
    "PerlinPyramidStairsTerrainCfg",
    "PerlinSlopeTerrainCfg",
    "PerlinSquareGapTerrainCfg",
    "PerlinStairsDownUpTerrainCfg",
    "PerlinStairsUpDownTerrainCfg",
    "PerlinSteppingStonesTerrainCfg",
    "PerlinTiltedRampTerrainCfg",
    "PerlinTiltTerrainCfg",
    "PerlinWaveTerrainCfg",
}

_TRIMESH_EXPORTS = {
    "MotionMatchedTerrainCfg",
    "STLHeightfieldTerrainCfg",
}

_VIRTUAL_OBSTACLE_EXPORTS = {
    "EdgeCylinderCfg",
    "FeatureEdgeCylinderCfg",
    "GreedyconcatEdgeCylinderCfg",
    "PluckerEdgeCylinderCfg",
    "RansacEdgeCylinderCfg",
    "RayEdgeCylinderCfg",
    "VirtualObstacleBase",
    "VirtualObstacleCfg",
}

__all__ = [
    "TerrainImporter",
    "TerrainImporterCfg",
    *_HEIGHT_FIELD_EXPORTS,
    *_TRIMESH_EXPORTS,
    *_VIRTUAL_OBSTACLE_EXPORTS,
]


def __getattr__(name: str):
    if name == "TerrainImporter":
        from .terrain_importer import TerrainImporter

        return TerrainImporter
    if name == "TerrainImporterCfg":
        from .terrain_importer_cfg import TerrainImporterCfg

        return TerrainImporterCfg
    if name in _HEIGHT_FIELD_EXPORTS:
        from .height_field import hf_terrains_cfg

        return getattr(hf_terrains_cfg, name)
    if name in _TRIMESH_EXPORTS:
        from .trimesh import mesh_terrains_cfg

        return getattr(mesh_terrains_cfg, name)
    if name in _VIRTUAL_OBSTACLE_EXPORTS:
        from . import virtual_obstacle

        return getattr(virtual_obstacle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
