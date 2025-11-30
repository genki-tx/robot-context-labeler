"""
Worker entry for per-episode processing.
Responsible for: preprocess -> Gemini call -> parse/validate -> write outputs.
Designed to be picklable and safe for multiprocessing.
"""

from dataclasses import asdict
from typing import Dict, Optional

from config import PipelineConfig, PromptConfig
from gemini_client import GeminiInferenceClient, build_prompt
from parser import parse_gemini_response
from preprocess import EpisodeMedia, preprocess_episode
from io_utils import write_episode_output, persist_raw

_CLIENT: Optional[GeminiInferenceClient] = None


def _get_client(model_name: str) -> GeminiInferenceClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GeminiInferenceClient(model_name=model_name)
    return _CLIENT


def process_episode(
    cfg: PipelineConfig,
    prompt_cfg: PromptConfig,
    chunk_id: str,
    episode_id: str,
    skip_gemini: bool,
) -> Dict:
    """
    Process a single episode. Returns a dict with status, warnings, errors, and output paths.
    """
    result = {
        "episode_id": episode_id,
        "chunk_id": chunk_id,
        "status": "pending",
        "warnings": [],
        "errors": [],
        "json_path": None,
        "raw_path": None,
    }

    media: Optional[EpisodeMedia] = preprocess_episode(
        dataset_root=cfg.dataset.root,
        chunk_id=chunk_id,
        episode_id=episode_id,
        cameras_cfg=cfg.cameras,
        processing_cfg=cfg.processing,
    )
    if not media:
        result["status"] = "preprocess_failed"
        result["warnings"].append("Preprocess failed or no frames.")
        return result

    if skip_gemini:
        result["status"] = "skipped_gemini"
        return result

    prompt_text = build_prompt(prompt_cfg, max_frame_count=media.frame_count)
    try:
        client = _get_client(cfg.gemini.model_name)
        raw_text = client.analyze_episode(media.video_path, prompt_text)
    except Exception as exc:
        result["status"] = "gemini_failed"
        result["errors"].append(f"Gemini call failed: {exc}")
        return result

    parsed = parse_gemini_response(episode_id, media.frame_count, raw_text)

    raw_path = persist_raw(cfg.output.dir, episode_id, raw_text)
    result["raw_path"] = raw_path

    if parsed.episode:
        json_path = write_episode_output(cfg.output, episode_id, parsed.episode)
        result["json_path"] = json_path
        result["status"] = "ok"
    else:
        result["status"] = "parse_failed"

    result["warnings"].extend(parsed.warnings)
    result["errors"].extend(parsed.errors)
    return result
