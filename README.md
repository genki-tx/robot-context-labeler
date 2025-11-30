# Gemini VLM Memory-Aware Labeler

Convert raw multi-camera LeRobot-style videos into **memory-aware** JSON labels using the Gemini API. The pipeline downsamples and stitches cameras with burned-in frame indices, calls Gemini to produce action segments plus skill scoring, validates the response, and writes per-episode JSON files.

## Why This Exists
- **Markov limitation**: Standard VLMs see only the current frame + instruction, so they “forget” past events. If a robot puts a red block in a drawer and closes it, the model thinks the block vanished.  
- **Memory awareness**: We inject textual memory-e.g., “The red block is inside the closed top drawer”-so models can reason about occluded/hidden state and object permanence.  
- **Scale problem**: Tens of thousands of episodes make manual temporal labeling impossible. We use Gemini to “watch” entire episodes, infer cause/effect, and emit memory-aware text.  
- **Quality signal**: In the same pass, Gemini scores execution quality (`skill_score` 1–3 with a short `skill_comment`) so we can prefer smooth/safe data, flag failures, and analyze failure modes without a separate pipeline.

## What You Get (Example Output)
For each episode, you get structured JSON capturing visible and inferred state plus a quality score. Example:
```json
{
  "episode_id": "000123",
  "overall_summary": "The robot approaches a red block, grasps it, places it in the top drawer, and closes it.",
  "skill_score": 2,
  "skill_comment": "Succeeded after a brief hesitation during grasp.",
  "segments": [
    {
      "start_frame": 1,
      "end_frame": 10,
      "action": "The robot moves toward the red block on the table.",
      "visual_state": "Red block visible on the table; gripper open.",
      "memory_context": "No objects are inside containers yet."
    },
    {
      "start_frame": 11,
      "end_frame": 20,
      "action": "The robot grasps the red block and lifts it.",
      "visual_state": "Red block held in the gripper.",
      "memory_context": "Red block is currently held."
    },
    {
      "start_frame": 21,
      "end_frame": 35,
      "action": "The robot places the red block into the top drawer and closes it.",
      "visual_state": "Drawer is closed; block no longer visible.",
      "memory_context": "Red block is inside the closed top drawer."
    }
  ]
}
```
This enables downstream training, filtering by skill quality, and reasoning over occluded objects.

## Pipeline Overview
1) **Config & Discovery**  
   - YAML-driven discovery under `<root>/videos/chunk-*/<camera>/episode_<id>.mp4`, respecting optional ID bounds.  
   - Multi-camera support: any set of camera directories can participate.
2) **Preprocess (“Burn-in & Stitch”)** (per camera per frame: **crop → resize → rotate → stitch**)  
   - Per-camera hooks: optional `crop` (x, y, w, h), optional `resize` (w, h) with warp (no aspect preservation), optional `rotate_deg` ∈ {0, 90, 180, 270}.  
   - FPS downsampling to reduce cost and emphasize state changes.  
   - Heights normalized when stitching; horizontal stitch gives a synchronized multi-view panorama.  
   - Burn `Frame: N` text on every stitched frame—this is the only timing source the model uses (1-indexed), eliminating hallucinated timestamps.  
   - Output written to `/dev/shm/gemini_VLM/episode_<id>.mp4` (optional debug copy on disk).
3) **Gemini Inference**  
   - Upload preprocessed video.  
   - Send a rules prompt (timing rules, segmentation guidelines, memory/object-permanence instructions, 1–3 skill scoring) plus dataset context from `prompt.yaml`.  
   - Model returns structured JSON text.
4) **Parse & Validate**  
   - Extract JSON, inject `episode_id`, enforce `skill_score` ∈ [1,3], clamp/sort frame bounds to burned indices, flag warnings.  
5) **Export**  
   - One JSON per episode; raw model response also saved for debugging.

## Prompt Contract (What Gemini Is Asked to Do)
- **Timing**: Use only the burned-in `Frame: N` text as the timing source; frames are 1-indexed. Ignore player duration/time bars.  
- **Segmentation**: Break into primitive actions (approach, grasp, in-hand manipulation, place/retreat, failures/idle) using the burned frame indices.  
- **Memory**: Track object permanence; if something goes into a container or becomes occluded, state where it is. Maintain recent actions in more detail, older in brief.  
- **Quality**: Assign `skill_score` in `[1,3]` with a concise `skill_comment` explaining the score.

## Code Structure
- `config/config.yaml`: Main pipeline config (dataset root, camera list, preprocessing, FPS, model, output pattern).  
- `config/prompt.yaml`: Base system prompt + dataset context concatenated into the final Gemini prompt.  
- `src/config.py`: YAML loaders → dataclasses; normalizes episode IDs and validates filename pattern.  
- `src/preprocess.py`: Frame hooks, downsample, stitch, burn timestamps, write tmp/debug videos.  
- `src/gemini_client.py`: Prompt assembly (rules + prompts), Gemini file upload + generate_content call.  
- `src/parser.py`: JSON extraction/validation and warning/error reporting.  
- `src/runner.py`: CLI orchestrator (discover → preprocess → Gemini → parse → export).  
- `src/gemini_vlm_quick_test.py`: Legacy proof-of-concept script (kept for reference).

## Requirements
- Python 3.10+  
- `pip install -e .` (installs `google-genai`)  
- Environment: `GEMINI_API_KEY` exported.  
- Dataset layout: `<dataset_root>/videos/chunk-*/<camera>/episode_<episode_id>.mp4`

## Configuration Highlights (`config/config.yaml`)
```yaml
dataset:
  root: "/dataset/example"        # path containing videos/
episodes:
  start_id: 1                     # inclusive; int or string; zero-padded internally
  end_id: 1
cameras:
  targets:
    - "observation.images.head_cam"
    - "observation.images.usb_cam"
  preprocessing:
    observation.images.head_cam:
      crop: [0, 0, 320, 480]      # x, y, width, height (optional)
    observation.images.usb_cam:
      resize: [640, 480]          # width, height (optional)
processing:
  target_fps: 1.0
  debug_keep_video: true
  debug_dir: "./debug_videos"
gemini:
  model_name: "gemini-3-pro-preview"  # use for production/best reasoning; gemini-2.0-flash-exp is cheaper for trial/debug
prompt_path: "./config/prompt.yaml"
output:
  dir: "./gemini_labels"
  filename_pattern: "episode_{episode_id}.json"  # must include {episode_id}
```

## Usage (Docker)
1) Prepare environment: copy and edit `.env` (set `GEMINI_API_KEY`, dataset paths, etc.):  
   ```bash
   cp example.env .env
   # edit .env to add your GEMINI_API_KEY and settings
   ```
2) Build and start the container:  
   ```bash
   docker compose build
   docker compose up -d
   docker exec -it context-labeler-<user> bash   # replace <user> with your compose profile/name
   ```
   The container auto-loads `.env`, so no manual `set -a` sourcing is needed.
3) Inside the container, adjust `config/config.yaml` for dataset root, episode range, cameras, and preprocessing.  
4) Run the pipeline:  
   ```bash
   python ./src/runner.py
   ```
5) Outputs:  
   - Labeled JSON: `./output_labels/episode_<id>.json`  
   - Raw model text: `./output_labels/episode_<id>_raw.txt`  
   - Debug stitched video (if enabled): `./debug_videos/episode_<id>.mp4`


## Image Preprocessing Tips
- **Resize**: `resize: [width, height]` warps the frame to exact dimensions (OpenCV does not preserve aspect ratio by default). Pick dimensions matching the source aspect, or crop first to your target aspect, then resize.  
- **Crop**: `crop: [x, y, width, height]` applies before resizing; use it to remove borders or enforce aspect ratios.  
- **Multi-camera stitching**: If cameras differ in size and you don’t resize, the stitcher normalizes heights proportionally; setting explicit `resize` per camera keeps widths predictable.  
- **Letterbox/padding**: Not implemented by default; add a padding step if you need to preserve aspect without warp.

## Troubleshooting
- **Missing key**: Runner logs a warning and skips Gemini; set `GEMINI_API_KEY`.  
- **No episodes found**: Check `dataset.root`, camera names, and `start_id/end_id` bounds.  
- **Stitching size issues**: Ensure preprocessing brings cameras to compatible heights (resize/crop).  
- **Model JSON errors**: Inspect `_raw.txt` and warnings; adjust prompt or validation if needed.

## Extending
- **Retry/Backoff**: Add retry logic in `GeminiInferenceClient.analyze_episode`.  
- **Additional preprocessing**: Extend `FramePreprocessingConfig` (color transforms, masks).  
- **Validation rules**: Tighten or relax constraints in `parser.py` (e.g., segment gap checks).  
- **Prompt variants**: Swap `prompt_path` per dataset or task; adjust `RULE_PROMPT` schema in `gemini_client.py`.  
- **JSONL aggregation**: Add a post-step to merge per-episode JSONs for training.
