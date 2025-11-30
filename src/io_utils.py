import json
import os
from typing import Dict


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
