"""
PROJECT: Gemini VLM API Memory Integrator (Proof of Concept)
-------------------------------------------------------------------
MOTIVATION & CONTEXT:
Modern Vision-Language-Action (VLA) models for robotics operate on a "Markov Assumption," meaning 
they typically only see the current frame and instruction. This makes them "amnesiac"—if a robot 
places an object into a drawer and closes it, the object disappears from the model's view, and the 
model forgets it exists. This limits the robot's ability to perform long-horizon tasks or retrieval.

THE SOLUTION:
We aim to train a new VLA foundation model that takes "Memory Context" as an input. To do this, 
we need to label our massive dataset (100k episodes) with text descriptions that explicitly track 
hidden states (e.g., "The red block is inside the closed top drawer").

THIS SCRIPT'S PURPOSE:
This is a "Golden Sample" validation script to verify the capabilities of the Google Gemini 3 
API before we build the full-scale processing pipeline.

IT VALIDATES TWO CRITICAL HYPOTHESES:
1. "Visual Timestamping" fixes Hallucinations: 
   Can the model accurately read burned-in text ("Frame: 12") to provide precise start/end times 
   for actions, avoiding the common issue where VLMs hallucinate timestamps?

2. "Object Permanence" Deduction: 
   Can the model watch a video where an object becomes hidden (e.g., put in a drawer) and 
   correctly output a 'memory_context' stating the object's location, even when it is no longer 
   visible in the pixels?

USAGE:
1. Place a short video (e.g., 'robot_drawer_test.mp4') in this directory.
2. Set your GEMINI_API_KEY environment variable.
3. Run: python gemini_vlm_quick_test.py
-------------------------------------------------------------------
"""

import cv2
import time
import os
import re
import json
from google import genai
from google.genai import types

# --- CONFIGURATION ---
INPUT_VIDEO = "/dataset/example/videos/chunk-000/observation.images.head_cam/episode_000001.mp4" # Replace with your test video path
INPUT_VIDEO = "/workspace/weblab_leader_unify_gripper_fake.mp4"
OUTPUT_VIDEO = "processed_for_gemini.mp4"
OUTPUT_JSON = "gemini_analysis.json"
API_KEY = os.environ.get("GEMINI_API_KEY") # Ensure this is set in your env
# Note: "gemini-2.0-flash-exp" is used for a cheaper and trial usage, use "gemini-3-pro-preview" for better reasoning.
MODEL_NAME = "gemini-3-pro-preview"
#MODEL_NAME = "gemini-2.0-flash-exp"

def preprocess_and_burn(input_path, output_path, target_fps=1.0):
    """
    Downsamples video and burns 'Frame: X' for the AI to read.
    Returns: The total number of frames in the processed video.
    """
    print(f"Loading {input_path}...")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate step to hit target FPS
    step = int(original_fps / target_fps)
    if step < 1: step = 1

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    processed_frame_count = 0
    read_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every Nth frame
        if read_frame_idx % step == 0:
            processed_frame_count += 1
            
            # --- BURN IN LOGIC ---
            # Text: Frame: 1, Frame: 2...
            label = f"Frame: {processed_frame_count}"
            
            # Draw black box background for contrast
            cv2.rectangle(frame, (10, height - 60), (300, height - 10), (0,0,0), -1)
            # Draw Green Text
            cv2.putText(frame, label, (20, height - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            out.write(frame)
            
        read_frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path} ({processed_frame_count} frames)")
    return processed_frame_count

def call_gemini_api(video_path):
    """
    Uploads video and asks for Memory-Aware Segmentation.
    """
    print("Initializing Gemini API...")
    client = genai.Client(api_key=API_KEY)

    print("Uploading video...")
    # Upload the video file
    video_file = client.files.upload(file=video_path)

    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    print("Video ready.")

    # --- THE PROMPT ---
    prompt = """
SYSTEM: You are a Robotics Perception Expert specializing in fine-grained manipulation analysis.
TASK: Analyze this video to generate a structured action log.

CRITICAL INSTRUCTION: 
1. Ignore the video player time. 
2. Look at the GREEN TEXT "Frame: X" at the bottom left. 
3. Use strictly those integers for your 'start_frame' and 'end_frame'.

OBJECTIVE:
Segment the video into primitive actions. Do not group complex behaviors; break them down into the following specific phases:
1. Approach/Start: The robot begins moving toward the target.
2. Pick/Grasp: The gripper makes contact and secures the item.
3. In-Hand Manipulation: The robot adjusts the posture or orientation of the item while grasping it.
4. Place: The robot lowers and releases the item.
5. Finish/Retreat: The robot moves away or returns to a neutral position.
This is example of actions. The robot may not do some of actions in the above, also may repeat actions, may fails. Describe what you found in the video.

MEMORY & CONTEXT INSTRUCTIONS:
The `memory_context` field is a dynamic state tracker. It must satisfy two conditions:
1. Object Permanence: If an object is placed inside a container or occluded, state: "The [object] is inside [container]" or "[object] is occluded."
2. Temporal Weighting (Short-term vs. Long-term): - Record the chronological behavior transition (what the robot did previously). 
   - Weighting Rule: Describe the *immediately preceding* action in detail. Summarize older actions briefly. Fade out/drop very old actions that are no longer relevant to the current state to simulate "forgetting."

OUTPUT FORMAT (JSON):
[
    {
        "start_frame": <int>,
        "end_frame": <int>,
        "action": "<Specific description of the current behivior>",
        "memory_context": "<weighted chronological log of past actions>"
    }
]
    """

    print("Sending request to model...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri,
                        mime_type=video_file.mime_type),
                    types.Part.from_text(text=prompt),
                ]
            )
        ]
    )
    
    return response.text

def extract_json_from_response(raw_response):
    """
    Cleans API response to extract strictly the JSON list part.
    Removes Markdown fences or conversational filler text.
    """
    match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if match:
        # Successfully extracted JSON from response
        return match.group(0)
    else:
        print("Warning: No JSON list found. Saving raw text.")
        return raw_response

if __name__ == "__main__":
    try:
        # 1. Preprocess
        total_frames = preprocess_and_burn(INPUT_VIDEO, OUTPUT_VIDEO)
        
        # 2. Call API
        t0 = time.time()
        raw_response = call_gemini_api(OUTPUT_VIDEO)
        process_duration = time.time() - t0
        
        print("\n--- Gemini API RESULTS ---\n")
        print(raw_response)
        print(f"API Process Time: {process_duration:.2f} seconds")

        # 3. Cleanup & Save
        json_result = extract_json_from_response(raw_response)

        with open(OUTPUT_JSON, "w") as f:
            f.write(json_result)
        print(f"Saved result to {OUTPUT_JSON}")

    except Exception as e:
        print(f"Error: {e}")