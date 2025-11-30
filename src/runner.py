import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from config import CamerasConfig, EpisodesConfig, PipelineConfig, load_config, load_prompt_config
from gemini_client import GeminiInferenceClient, build_prompt
from parser import parse_gemini_response
from preprocess import EpisodeMedia, preprocess_episode


def _episode_id_from_filename(path: str) -> Optional[str]:
    match = re.search(r"episode_(\d+)\.mp4$", os.path.basename(path))
    return match.group(1) if match else None


def _within_range(episode_id: str, episodes_cfg: EpisodesConfig) -> bool:
    start_ok = episodes_cfg.start_id is None or episode_id >= episodes_cfg.start_id
    end_ok = episodes_cfg.end_id is None or episode_id <= episodes_cfg.end_id
    return start_ok and end_ok


def discover_episodes(
    dataset_root: str, cameras_cfg: CamerasConfig, episodes_cfg: EpisodesConfig
) -> List[Tuple[str, str]]:
    videos_root = os.path.join(dataset_root, "videos")
    chunk_dirs = sorted(d for d in glob.glob(os.path.join(videos_root, "chunk-*")) if os.path.isdir(d))
    discovered: List[Tuple[str, str]] = []

    for chunk_path in chunk_dirs:
        chunk_id = os.path.basename(chunk_path)
        per_camera_ids: List[set] = []
        for camera in cameras_cfg.targets:
            camera_dir = os.path.join(chunk_path, camera)
            if not os.path.isdir(camera_dir):
                print(f"[WARN] Camera directory missing: {camera_dir}")
                per_camera_ids.append(set())
                continue
            ids = {_episode_id_from_filename(p) for p in glob.glob(os.path.join(camera_dir, "episode_*.mp4"))}
            per_camera_ids.append({i for i in ids if i})

        if not per_camera_ids:
            continue
        candidates = set.intersection(*per_camera_ids) if len(per_camera_ids) > 1 else per_camera_ids[0]
        for episode_id in sorted(candidates):
            if _within_range(episode_id, episodes_cfg):
                discovered.append((chunk_id, episode_id))

    return discovered


def write_episode_output(output_cfg, episode_id: str, payload: Dict) -> str:
    os.makedirs(output_cfg.dir, exist_ok=True)
    filename = output_cfg.filename_pattern.format(episode_id=episode_id)
    path = os.path.join(output_cfg.dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True)
    return path


def persist_raw(output_dir: str, episode_id: str, raw_text: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"episode_{episode_id}_raw.txt")
    with open(path, "w") as f:
        f.write(raw_text)
    return path


def run_pipeline(cfg: PipelineConfig, args):
    episodes = discover_episodes(cfg.dataset.root, cfg.cameras, cfg.episodes)

    if not episodes:
        print("[INFO] No episodes discovered. Check dataset root and camera targets.")
        return

    prompt_cfg = load_prompt_config(cfg.prompt_path)
    client = None
    skip_gemini = args.skip_gemini
    if not skip_gemini:
        try:
            client = GeminiInferenceClient(cfg.gemini.model_name)
        except ValueError as exc:
            print(f"[WARN] {exc}. Set GEMINI_API_KEY or use --skip-gemini. Falling back to skip.")
            skip_gemini = True

    for chunk_id, episode_id in episodes:
        print(f"[INFO] Processing {chunk_id} / episode_{episode_id}")
        media: Optional[EpisodeMedia] = preprocess_episode(
            dataset_root=cfg.dataset.root,
            chunk_id=chunk_id,
            episode_id=episode_id,
            cameras_cfg=cfg.cameras,
            processing_cfg=cfg.processing,
        )
        if not media:
            print(f"[WARN] Skipping episode {episode_id} (preprocess failed).")
            continue

        if skip_gemini:
            print(f"[INFO] Skipping Gemini call for episode {episode_id}.")
            continue

        prompt_text = build_prompt(prompt_cfg, max_frame_count=media.frame_count)
        raw_text = client.analyze_episode(media.video_path, prompt_text)
        parsed = parse_gemini_response(episode_id, media.frame_count, raw_text)

        if parsed.episode:
            out_path = write_episode_output(cfg.output, episode_id, parsed.episode)
            print(f"[INFO] Wrote episode JSON to {out_path}")
        else:
            print(f"[WARN] Parsing failed for episode {episode_id}. See raw output.")

        raw_path = persist_raw(cfg.output.dir, episode_id, raw_text)
        if parsed.warnings:
            print(f"[WARN] Episode {episode_id} warnings:")
            for w in parsed.warnings:
                print(f" - {w}")
        if parsed.errors:
            print(f"[ERROR] Episode {episode_id} errors:")
            for e in parsed.errors:
                print(f" - {e}")
        print(f"[DEBUG] Raw model response saved to {raw_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini VLM memory-aware labeling pipeline.")
    parser.add_argument("--config", default="./config/config.yaml", help="Path to pipeline YAML config.")
    parser.add_argument("--skip-gemini", action="store_true", help="Run preprocessing only, skip Gemini calls.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    config = load_config(arguments.config)
    run_pipeline(config, arguments)
