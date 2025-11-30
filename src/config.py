from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class FramePreprocessingConfig:
    crop: Optional[List[int]] = None  # [x, y, width, height]
    resize: Optional[List[int]] = None  # [width, height]
    rotate_deg: Optional[int] = None  # rotation in degrees (applied after crop/resize)


@dataclass
class DatasetConfig:
    root: str


@dataclass
class EpisodesConfig:
    start_id: Optional[str] = None
    end_id: Optional[str] = None


@dataclass
class CamerasConfig:
    targets: List[str]
    preprocessing: Dict[str, FramePreprocessingConfig] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    target_fps: float = 1.0
    debug_keep_video: bool = False
    debug_dir: str = "./debug_videos"


@dataclass
class GeminiConfig:
    model_name: str


@dataclass
class OutputConfig:
    dir: str
    filename_pattern: str


@dataclass
class PromptConfig:
    base_system_prompt: str
    dataset_specific_context: str


@dataclass
class PipelineConfig:
    dataset: DatasetConfig
    episodes: EpisodesConfig
    cameras: CamerasConfig
    processing: ProcessingConfig
    gemini: GeminiConfig
    output: OutputConfig
    prompt_path: str


def _build_frame_preprocessing(raw_cfg: Dict[str, Dict]) -> Dict[str, FramePreprocessingConfig]:
    result: Dict[str, FramePreprocessingConfig] = {}
    for camera, cfg in (raw_cfg or {}).items():
        result[camera] = FramePreprocessingConfig(
            crop=cfg.get("crop"),
            resize=cfg.get("resize"),
            rotate_deg=cfg.get("rotate_deg"),
        )
    return result


def load_config(path: str) -> PipelineConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Configuration file is empty.")

    dataset_cfg = DatasetConfig(root=raw["dataset"]["root"])
    def _normalize_episode_id(value):
        if value is None:
            return None
        if isinstance(value, int):
            return f"{value:06d}"
        if isinstance(value, str):
            return value.zfill(6) if value.isdigit() else value
        raise ValueError(f"episodes IDs must be str or int, got {type(value)}")

    episodes_cfg = EpisodesConfig(
        start_id=_normalize_episode_id(raw.get("episodes", {}).get("start_id")),
        end_id=_normalize_episode_id(raw.get("episodes", {}).get("end_id")),
    )
    cameras_raw = raw.get("cameras", {})
    if "targets" not in cameras_raw or not cameras_raw["targets"]:
        raise ValueError("cameras.targets must list at least one camera.")
    cameras_cfg = CamerasConfig(
        targets=cameras_raw["targets"],
        preprocessing=_build_frame_preprocessing(cameras_raw.get("preprocessing", {})),
    )
    processing_raw = raw.get("processing", {})
    processing_cfg = ProcessingConfig(
        target_fps=processing_raw.get("target_fps", 1.0),
        debug_keep_video=processing_raw.get("debug_keep_video", False),
        debug_dir=processing_raw.get("debug_dir", "./debug_videos"),
    )
    gemini_cfg = GeminiConfig(model_name=raw["gemini"]["model_name"])
    output_raw = raw["output"]
    if "{episode_id}" not in output_raw["filename_pattern"]:
        raise ValueError("output.filename_pattern must include '{episode_id}'.")
    output_cfg = OutputConfig(
        dir=output_raw["dir"],
        filename_pattern=output_raw["filename_pattern"],
    )
    prompt_path = raw.get("prompt_path", "./config/prompt.yaml")

    return PipelineConfig(
        dataset=dataset_cfg,
        episodes=episodes_cfg,
        cameras=cameras_cfg,
        processing=processing_cfg,
        gemini=gemini_cfg,
        output=output_cfg,
        prompt_path=prompt_path,
    )


def load_prompt_config(path: str) -> PromptConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return PromptConfig(
        base_system_prompt=raw.get("base_system_prompt", "").strip(),
        dataset_specific_context=raw.get("dataset_specific_context", "").strip(),
    )
