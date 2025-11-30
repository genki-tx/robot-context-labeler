import os
import time
from typing import Optional

from google import genai
from google.genai import types

from config import PromptConfig

RULE_PROMPT = """
SYSTEM: You are a Robotics Perception Expert specializing in fine-grained manipulation analysis.
TASK: Analyze this video and generate a structured action log with memory-awareness and task quality assessment.

TIMING RULES:
- Ignore the video player's duration or time bar.
- Use only the burned-in text "Frame: X" at the bottom of the video.
- Use these integers directly for `start_frame` and `end_frame`.

SEGMENTATION:
- Segment the episode into primitive actions (approach, grasp, in-hand manipulation, place, retreat, failure sequences, etc.).
- Explicitly describe any failures (drops, missed grasps, getting stuck, long idle pauses, human intervention).

MEMORY & CONTEXT:
- Track object permanence: if an object is placed inside a container or becomes occluded, state where it is (e.g., "The red block is inside the closed top drawer.").
- Maintain a short “rolling memory”: describe recent actions in more detail and summarize older, no-longer-relevant actions.

TASK QUALITY ASSESSMENT:
- Assign a `skill_score` from 1 to 3 for the overall episode:
  - 3: Excellent or "Golden Data" — succeeds on first attempt, direct path, no noticeable collisions or stalls.
  - 2: Recovered — succeeds but requires retries, recovery, or shows inefficiency.
  - 1: Failure — task not completed or unsafe behavior (drops unrecovered objects, major collisions, human intervention).
- Provide a short `skill_comment` explaining why you chose this score.

OUTPUT FORMAT:
- Return strictly valid JSON with the schema described below. Do NOT include Markdown code fences or any extra commentary.
""".strip()


def build_prompt(prompt_cfg: PromptConfig, max_frame_count: Optional[int]) -> str:
    schema_prompt = """
Expected JSON schema:
{
  "overall_summary": "Short natural language summary of the entire episode.",
  "skill_score": 1,
  "skill_comment": "Why this score was chosen.",
  "segments": [
    {
      "start_frame": 12,
      "end_frame": 20,
      "action": "Detailed description of what the robot does and to what object.",
      "visual_state": "What is visibly present.",
      "memory_context": "What must be inferred from past events (object permanence)."
    }
  ]
}
Rules:
- `skill_score` must be an integer between 1 and 3.
- `start_frame` and `end_frame` must be integers based on the burned-in frame index.
- Ensure 1 <= start_frame <= end_frame.
- Prefer non-empty segments; include failures or idle periods as explicit segments.
""".strip()

    frame_note = ""
    if max_frame_count:
        frame_note = f"\nMax valid frame index for this episode: {max_frame_count}."

    parts = [
        RULE_PROMPT,
        prompt_cfg.base_system_prompt,
        "DATASET CONTEXT:",
        prompt_cfg.dataset_specific_context,
        schema_prompt + frame_note,
    ]
    return "\n\n".join(part.strip() for part in parts if part)


class GeminiInferenceClient:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to call the Gemini API.")
        self.client = genai.Client(api_key=self.api_key)

    def analyze_episode(self, video_path: str, prompt_text: str) -> str:
        start_time = time.time()
        video_file = self.client.files.upload(file=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(1.0)
            video_file = self.client.files.get(name=video_file.name)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
                        types.Part.from_text(text=prompt_text),
                    ],
                )
            ],
        )
        duration = time.time() - start_time
        print(f"[INFO] Gemini call completed in {duration:.2f}s for {video_path}")
        return response.text
