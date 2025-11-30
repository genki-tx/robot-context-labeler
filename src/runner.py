import argparse
import glob
import os
import re
import concurrent.futures
from typing import Dict, List, Optional, Tuple

from config import CamerasConfig, EpisodesConfig, PipelineConfig, load_config, load_prompt_config
from gemini_client import GeminiInferenceClient
from worker import process_episode


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


def run_pipeline(cfg: PipelineConfig, args):
    episodes = discover_episodes(cfg.dataset.root, cfg.cameras, cfg.episodes)

    if not episodes:
        print("[INFO] No episodes discovered. Check dataset root and camera targets.")
        return

    prompt_cfg = load_prompt_config(cfg.prompt_path)
    skip_gemini = args.skip_gemini
    if not skip_gemini:
        try:
            GeminiInferenceClient(cfg.gemini.model_name)
        except ValueError as exc:
            print(f"[WARN] {exc}. Set GEMINI_API_KEY or use --skip-gemini. Falling back to skip.")
            skip_gemini = True

    workers = max(1, int(getattr(cfg.processing, "workers", 1)))

    if workers > 1:
        print(f"[INFO] Running with {workers} workers.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_episode, cfg, prompt_cfg, chunk_id, episode_id, skip_gemini)
                for chunk_id, episode_id in episodes
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"episode_id": "unknown", "status": "error", "errors": [str(exc)]}
                _log_result(res)
    else:
        for chunk_id, episode_id in episodes:
            res = process_episode(cfg, prompt_cfg, chunk_id, episode_id, skip_gemini)
            _log_result(res)


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini VLM memory-aware labeling pipeline.")
    parser.add_argument("--config", default="./config/config.yaml", help="Path to pipeline YAML config.")
    parser.add_argument("--skip-gemini", action="store_true", help="Run preprocessing only, skip Gemini calls.")
    return parser.parse_args()


def _log_result(res: Dict):
    episode_id = res.get("episode_id", "?")
    status = res.get("status", "unknown")
    print(f"[INFO] Episode {episode_id} status: {status}")
    if res.get("json_path"):
        print(f"[INFO] JSON: {res['json_path']}")
    if res.get("raw_path"):
        print(f"[DEBUG] Raw: {res['raw_path']}")
    for w in res.get("warnings", []):
        print(f"[WARN] {episode_id}: {w}")
    for e in res.get("errors", []):
        print(f"[ERROR] {episode_id}: {e}")


if __name__ == "__main__":
    arguments = parse_args()
    config = load_config(arguments.config)
    run_pipeline(config, arguments)
