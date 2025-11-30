# **Project Blueprint for Gemini VLM Integrator**

## **1. Mission Statement**

We are building a data preprocessing pipeline for Robotics AI.
**Goal:** Convert raw multi-camera video logs (LeRobot-style format) into **“Memory-Aware”** text descriptions using the **Gemini API**.
**Outcome:** A JSON object for each robot episode that:

* Segments the video into action segments with integer frame indices, and
* Describes both **visible state** and **hidden state** (e.g., objects inside closed containers), and
* Assigns a **task quality score** (skill_score 1-3) with a short explanation.

This will be used to train a smarter VLM model that has both **memory** and a notion of **execution quality.**

---

## **1.1 Strategic Context & Motivation**

### Why are we building this?

Current Foundation Models for Robotics (VLM – Vision-Language-Action) suffer from a critical limitation known as the **Markov Assumption**.

* **The Problem:**
  Model inputs are typically just the *current* image frame and the *current* instruction. This makes the robot **“amnesiac”** – it has no memory of what happened 10 seconds ago.

  * *Example:* If a robot puts a red block in a drawer and closes it, the block disappears from view. A standard VLM model now effectively believes the block is gone. It cannot reliably answer “Where is the red block?” or “Retrieve the red block.”

* **The Solution (“Memory-Awareness”):**
  We want to inject “memory” into the model via text. By providing a text description like

  > *"The red block is inside the closed top drawer."*
  > alongside the current image, we allow the robot to “see” the hidden past and reason over **object permanence** and **occluded state**.

* **The Data Challenge:**
  We have a huge amount of robot data (tens of thousands of episodes).

  * It is impossible for humans to watch and label temporal context for all these videos.
  * We need an automated agent (Gemini API) to “watch” full episodes, understand cause-and-effect over time, and generate these “Memory Context” labels.

### Why do we also care about task quality (skill_score)?

Not all episodes are equal:

* Some episodes show **smooth, efficient, safe** behavior (good for imitation learning).
* Some episodes contain:

  * Failed grasps, dropped objects, getting stuck,
  * Long idle waits, clumsy motion,
  * Collisions or dangerous behaviors,
  * Human intervention to unblock the task.

We want the same Gemini pass to also produce:

* A **skill_score** (1–3) describing how good the execution looks within that episode, and
* A short **skill_comment** explaining why (e.g., “dropped the object twice”, “smooth single-shot grasp and place”).

This enables us to:

* Prefer high-skill episodes for training,
* Tag anomalous or unsafe episodes,
* Analyze failure modes without a separate labeling pipeline.

---

## **2. Architecture Overview**

Per episode, the pipeline has 4 stages:

1. **Config Loader & Episode Discovery**
2. **Preprocessor (“Burn-in & Stitch”)**
3. **Gemini Inference Engine (Gemini API)**
4. **Parser, Validation & JSON Export**

Configuration is driven by a **YAML file**.

---

## **3. YAML Configuration**

We assume a **fixed LeRobot-style layout** for videos:

> `<dataset_root>/videos/<chunk-id>/<camera-id>/<episode-id>.mp4`
> e.g. `example/videos/chunk-000/observation.images.head_cam/episode_000000.mp4`

### 3.1 YAML Schema (Conceptual)

```yaml
dataset:
  # Root path of the dataset (points to repo/dataset root that contains "videos/")
  root: "/dataset/example"

episodes:
  # Inclusive episode ID range, using zero-padded IDs in filenames.
  # If omitted, process all discovered episodes.
  start_id: "000005"   # optional
  end_id: "000010"     # optional

cameras:
  # List of camera directories (relative to "<root>/videos/<chunk-id>/")
  # Example:
  #   example/videos/chunk-000/observation.images.head_cam/episode_000000.mp4
  #   example/videos/chunk-000/observation.images.usb_cam/episode_000000.mp4
  targets:
    - "observation.images.head_cam"
    - "observation.images.usb_cam"

  # Optional per-camera image preprocessing (ready for future use)
  preprocessing:
    observation.images.head_cam:
      # crop: [x, y, width, height]   # optional
      # resize: [640, 480]            # optional
    observation.images.usb_cam:
      # resize: [640, 480]

processing:
  target_fps: 1.0             # downsampled FPS for burned video
  debug_keep_video: false     # if true, keep preprocessed videos on disk for debugging
  debug_dir: "./debug_videos" # where to copy debug videos if debug_keep_video is true

gemini:
  model_name: "gemini-3-pro-preview"

prompt_path: "./config/prompt.yaml"

output:
  dir: "./gemini_labels"
  filename_pattern: "episode_{episode_id}.json"
```

### 3.2 Behavior from YAML

* **dataset.root**

  * Code assumes videos live under: `root/videos/`.
  * Scans `root/videos/chunk-*/<camera>/<episode_*.mp4>` to discover episodes.

* **episodes.start_id / end_id**

  * If specified: only process episodes where `start_id <= episode_id <= end_id`.
  * If omitted: process all discovered episodes.

* **cameras.targets**

  * Defines which camera streams participate in stitching and in what order.

* **cameras.preprocessing**

  * Per-camera crop/resize hooks (optional for now, but pipeline must call the functions).

* **processing.*:**

  * `target_fps` controls temporal downsampling.
  * Preprocessed video is always written to a fixed **tmp path** under `/dev/shm/gemini_VLM/`.
  * If `debug_keep_video: true`, the tmp file is copied or moved into `debug_dir`.

* **gemini.model_name:**

  * Model to use with the Gemini API.

* **output.*:**

  * `dir`: where per-episode JSON files are written.
  * `filename_pattern`: must contain `{episode_id}` placeholder.

### Prompt Files (External YAML)

To keep the main configuration file clean and to allow easy modification of long prompt text, all prompt strings are saved in a separate YAML file.

* prompt.yaml (Example Structure)
  * `base_system_prompt` + `dataset_specific_context` are concatenated into the final prompt.
```
base_system_prompt: |
  You are a Robotics Perception Expert who analyzes robot manipulation videos...

dataset_specific_context: |
  # Optional: Customize per dataset
  This dataset is recorded in a tabletop manipulation environment...
```

---

## **4. Stage 1: Preprocessor (“Burn-in & Stitch”)**

### 4.1 Inputs

For a given `episode_id`:

* `dataset.root`
* Discovered `chunk-id` (or iterate over all chunks)
* `cameras.targets` list (camera directories)

### 4.2 Processing Steps (Modular)

The code should be structured as small, composable functions:

1. **Episode Video Discovery**

   * For each camera in `cameras.targets`:

     * Build path:
       `root/videos/<chunk-id>/<camera>/episode_{episode_id}.mp4`
     * Open with `cv2.VideoCapture`.
   * If any required video is missing → log and skip episode (or mark error).

2. **Per-Camera Frame Preprocessing (hook)**

   * For each read frame:

     * Apply per-camera crop/resize if configured in `cameras.preprocessing`.
   * Suggested interface:

     ```python
     def preprocess_frame(frame, camera_id, camera_cfg):
         # crop / resize / color conversions
         return frame
     ```

3. **Temporal Downsampling**

   * Assume all cameras share the same FPS (take FPS from first video).
   * Compute:

     ```python
     step = int(original_fps / target_fps)
     step = max(step, 1)
     ```
   * Read frames sequentially; only process frames where `read_frame_idx % step == 0`.

4. **Multi-Camera Stitching**

   * For each kept timestamp:

     * Read one frame per camera.
     * Apply `preprocess_frame`.
     * Stack horizontally: `stitched = np.hstack([...])`.

5. **Visual Timestamp Burn-in**

   * Maintain `processed_frame_count` starting at 1.
   * For each stitched frame:

     * Draw a dark rectangle at the bottom (for contrast).
     * Overlay bright text: `"Frame: {processed_frame_count}"`.
   * This integer index is the *only* “time” we rely on.

6. **Write Output Video to /dev/shm**

   * Use `cv2.VideoWriter` with:

     * fourcc: `'mp4v'` (fixed in code)
     * fps: `target_fps`
     * frame size: stitched width, height
   * Path example:
     `/dev/shm/gemini_VLM/episode_{episode_id}.mp4`

7. **Debug Copy (optional)**

   * If `debug_keep_video: true`:

     * Copy the `/dev/shm` file into `debug_dir`.

### 4.3 Outputs

* Preprocessed video file path:
  `/dev/shm/gemini_VLM/episode_{episode_id}.mp4`
* `processed_frame_count` (maximum valid frame number)

This stage is purely video I/O + geometry. It has no Gemini or JSON knowledge.

---

## **5. Stage 2: Gemini Inference Engine**

### 5.1 Inputs

* Preprocessed video path for the episode
* `gemini.model_name`
* Combined prompt built from:

  * Fixed “rules” prompt (timing, segmentation, memory, skill scoring)
  * `prompt.base_system_prompt`
  * `prompt.dataset_specific_context`

### 5.2 Prompt Strategy (Conceptual Template)
Refer `./config/prompt.yaml`.

We then append the explicit schema definition in the text.

### 5.3 Gemini Call (Runtime)

* Use environment variable `GEMINI_API_KEY` (fixed name, no YAML field).
* Use `google-genai` client:

  * `client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])`
  * Upload video file as a Gemini file resource.
  * Call `models.generate_content` with:

    * A `file` part for the video
    * A `text` part containing the prompt

Basic retry can be added later; for now the blueprint assumes a single call and error handling around exceptions.

---

## **6. Stage 3: Parser & Validator**

### 6.1 Target JSON Schema (Per Episode)

```json
{
  "episode_id": "000123",
  "overall_summary": "Short natural language summary of the entire episode.",
  "skill_score": 1,
  "skill_comment": "The robot completed the task but dropped the object once and paused for several seconds before retrying.",
  "segments": [
    {
      "start_frame": 12,
      "end_frame": 20,
      "action": "The robot approaches the red cube on the table.",
      "visual_state": "Red cube visible on the table; gripper is open.",
      "memory_context": "No objects are currently inside containers."
    },
    {
      "start_frame": 21,
      "end_frame": 35,
      "action": "The robot grasps the red cube and lifts it.",
      "visual_state": "Red cube visible in the gripper.",
      "memory_context": "The red cube is currently held in the gripper."
    },
    {
      "start_frame": 36,
      "end_frame": 60,
      "action": "The robot places the red cube inside the top drawer and closes it.",
      "visual_state": "Drawer front is visible and closed; red cube is no longer visible.",
      "memory_context": "The red cube is inside the closed top drawer."
    }
  ]
}
```

### 6.2 Parsing

* The raw Gemini response may include some natural language before/after JSON.
* Implement a helper:

  * Try to parse the whole response as JSON.
  * If it fails, use a regex to extract the largest `{ ... }` block and parse that.
* If parsing fails completely:

  * Save the raw response for debugging and mark the episode as failed.

### 6.3 Validation Rules

* `episode_id`:

  * Must match the episode being processed (from filename). If absent, inject it from code.
  * Gemini may not insert this ID to json, add or correct it in the validation and correction step.

* `skill_score`:

  * Must be an integer in [1, 3].
  * If missing or out of range, log a warning and set to `null` or a sentinel.

* `segments`:

  * Prefer non-empty list, but allow empty with a warning.
  * For each segment:

    * `start_frame` and `end_frame` must be integers.
    * `1 <= start_frame <= end_frame <= processed_frame_count`.
  * Global monotonicity:

    * For `i > 0`: `segments[i].start_frame >= segments[i-1].end_frame`.
    * If violated: either sort and clip with warnings or mark the episode as “needs review”.

If critical checks fail (e.g., all frames out of range), mark the episode as failed and optionally write a small error log, and leave the raw output there.

---

## **7. Stage 4: Episode Export (JSON)**

### 7.1 Per-Episode JSON

For each successfully parsed and validated episode:

* Build a Python dict matching the schema above (ensuring `episode_id` is set).
* Write to file:

  * Path: `{output.dir}/{filename_pattern.format(episode_id=episode_id)}`
    e.g. `./gemini_labels/episode_000123.json`
  * Only one line per file (one JSON object for that episode).


---

If you’d like, next step I can take this blueprint and refactor your prototype script into:

* `config.py` (YAML loader + dataclasses)
* `preprocess.py` (Stage 1)
* `gemini_client.py` (Stage 2)
* `parser.py` (Stage 3)
* `runner.py` (multiprocessing loop over episodes)

## **8. Example code**
We already implemented prototype code at `src/gemini_vlm_quick_test.py`. Refer it for the starting point.