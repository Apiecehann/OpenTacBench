"""Shared scene assets and defaults for both benchmark and standalone tasks."""
import os
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from ._global import SCENE_ASSETS_ROOT


def resolve_scene_background_texture(background: str | os.PathLike | None) -> Path:
    text = "base5" if background is None else str(background).strip()
    if not text: text = "base5"
    candidate = Path(text)
    if candidate.suffix != ".exr":
        candidate = SCENE_ASSETS_ROOT / f"{text}.exr"
    elif not candidate.is_absolute():
        candidate = SCENE_ASSETS_ROOT / candidate
    if not candidate.exists():
        available = ", ".join(sorted(path.stem for path in SCENE_ASSETS_ROOT.glob("base*.exr")))
        raise FileNotFoundError(f"Scene background {text!r} not found at {candidate}. Available backgrounds: {available}")
    return candidate


def scene_light_cfg(prim_path='/World/light',background='base5'):
    return AssetBaseCfg(prim_path=prim_path,spawn=sim_utils.DomeLightCfg(
        color=(.75,.75,.75),intensity=1500.,
        texture_file=str(SCENE_ASSETS_ROOT / f'{background}.exr')))


def scene_plate_cfg(prim_path='/World/envs/env_.*/ground_plate'):
    return RigidObjectCfg(prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.5,0,0)),
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_ASSETS_ROOT / 'plate01.usda'),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,solver_velocity_iteration_count=1,
                max_angular_velocity=1000.,max_linear_velocity=1000.,
                max_depenetration_velocity=5.,kinematic_enabled=True)))
