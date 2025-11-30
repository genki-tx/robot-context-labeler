import os
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from config import CamerasConfig, FramePreprocessingConfig, ProcessingConfig


@dataclass
class EpisodeMedia:
    chunk_id: str
    episode_id: str
    video_path: str
    frame_count: int


def preprocess_frame(frame, camera_id: str, preprocessing: Dict[str, FramePreprocessingConfig]):
    cfg: Optional[FramePreprocessingConfig] = preprocessing.get(camera_id)
    if not cfg:
        return frame

    processed = frame
    if cfg.crop:
        x, y, w, h = cfg.crop
        x = max(int(x), 0)
        y = max(int(y), 0)
        w = max(int(w), 0)
        h = max(int(h), 0)
        if w > 0 and h > 0:
            processed = processed[y : y + h, x : x + w]
    if cfg.resize:
        width, height = cfg.resize
        processed = cv2.resize(processed, (int(width), int(height)))
    return processed


def _normalize_heights(frames):
    target_height = frames[0].shape[0]
    normalized = []
    for frame in frames:
        if frame.shape[0] != target_height:
            scale = target_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_width, target_height))
        normalized.append(frame)
    return normalized


def _burn_timestamp(frame, frame_index: int):
    height, width, _ = frame.shape
    label = f"Frame: {frame_index}"
    box_height = 40
    cv2.rectangle(frame, (0, height - box_height), (width, height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    return frame


def preprocess_episode(
    dataset_root: str,
    chunk_id: str,
    episode_id: str,
    cameras_cfg: CamerasConfig,
    processing_cfg: ProcessingConfig,
) -> Optional[EpisodeMedia]:
    camera_paths = {
        camera: os.path.join(
            dataset_root,
            "videos",
            chunk_id,
            camera,
            f"episode_{episode_id}.mp4",
        )
        for camera in cameras_cfg.targets
    }

    for path in camera_paths.values():
        if not os.path.exists(path):
            print(f"[WARN] Missing video for episode {episode_id}: {path}")
            return None

    caps = {camera: cv2.VideoCapture(path) for camera, path in camera_paths.items()}
    for camera, cap in caps.items():
        if not cap.isOpened():
            print(f"[WARN] Could not open video for {camera} ({camera_paths[camera]}).")
            return None

    first_cap = next(iter(caps.values()))
    original_fps = first_cap.get(cv2.CAP_PROP_FPS) or 1.0
    step = max(int(original_fps / processing_cfg.target_fps), 1)

    tmp_dir = "/dev/shm/gemini_VLM"
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = os.path.join(tmp_dir, f"episode_{episode_id}.mp4")

    writer = None
    processed_frame_count = 0
    read_frame_idx = 0

    while True:
        frames = []
        for camera, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                frames = []
                break
            frames.append(preprocess_frame(frame, camera, cameras_cfg.preprocessing))

        if not frames:
            break

        if read_frame_idx % step != 0:
            read_frame_idx += 1
            continue

        frames = _normalize_heights(frames)
        stitched = np.hstack(frames)
        stitched = _burn_timestamp(stitched, processed_frame_count + 1)

        if writer is None:
            height, width, _ = stitched.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, processing_cfg.target_fps, (width, height))

        writer.write(stitched)
        processed_frame_count += 1
        read_frame_idx += 1

    for cap in caps.values():
        cap.release()
    if writer:
        writer.release()

    if processed_frame_count == 0:
        print(f"[WARN] No frames processed for episode {episode_id}.")
        return None

    if processing_cfg.debug_keep_video:
        os.makedirs(processing_cfg.debug_dir, exist_ok=True)
        debug_path = os.path.join(processing_cfg.debug_dir, f"episode_{episode_id}.mp4")
        shutil.copy(output_path, debug_path)
        print(f"[DEBUG] Copied debug video to {debug_path}")

    return EpisodeMedia(
        chunk_id=chunk_id,
        episode_id=episode_id,
        video_path=output_path,
        frame_count=processed_frame_count,
    )
