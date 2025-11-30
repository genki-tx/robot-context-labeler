### **1. Cost Structure (Gemini 3 Pro Preview)**

The model uses a tiered pricing model based on the length of your prompt (Context Window).

| Item | **Standard Context** (≤ 200k tokens) | **Long Context** (\> 200k tokens) |
| :--- | :--- | :--- |
| **Input Price** (Video + Text) | **$2.00** / 1 million tokens | **$4.00** / 1 million tokens |
| **Output Price** (Model Response) | **$12.00** / 1 million tokens | **$18.00** / 1 million tokens |

> **Note:**
>
>   * **File API Storage (`client.files.upload`):** **Free.** You are not charged for uploading or storing the video for the standard 48-hour retention period. You only pay for the *tokens* when you send the video to the model.
>   * **Context Caching:** If you use the explicit *Context Caching* feature (to query the same video many times), there is a separate storage charge (\~$4.50/1M tokens/hour).

-----

### **2. Video Tokenization (The "Gemini 3" Difference)**

Gemini 3 Pro introduces granular control over video cost using the `media_resolution` parameter. It is much cheaper per frame than Gemini 1.5.

  * **Token Count:**
      * **Low/Medium Resolution (Default):** **70 tokens per frame** (VGA falls here).
      * **High Resolution:** 280 tokens per frame.
  * **Audio:** \~32 tokens per second (if present).
  * **Sampling Rate:** By default, the API samples video at **1 frame per second (FPS)**, regardless of the file's native frame rate (e.g., 10fps).

-----

### **3. Cost Calculation for Your Scenario**

**Scenario:** 10fps VGA video, 100 seconds long, no audio.

#### **A. Default Behavior (Most Likely)**

Unless you write code to force the API to process every single frame, it will sample at **1 FPS**.

  * **Duration:** 100 seconds
  * **Sampled Frames:** 100 frames (1 frame/sec)
  * **Tokens:** $100 \times 70 = \mathbf{7,000 \text{ tokens}}$
  * **Tier:** ≤ 200k (Standard)
  * **Estimated Cost:**
    $$\frac{7,000}{1,000,000} \times \$2.00 = \mathbf{\$0.014 \text{ (1.4 cents)}}$$

#### **B. "Native" Processing (Forcing 10 FPS)**

If you use the `frame_rate` option or extract frames to force 10 FPS analysis:

  * **Sampled Frames:** 1,000 frames (100 sec × 10 fps)
  * **Tokens:** $1,000 \times 70 = \mathbf{70,000 \text{ tokens}}$
  * **Estimated Cost:**
    $$\frac{70,000}{1,000,000} \times \$2.00 = \mathbf{\$0.14 \text{ (14 cents)}}$$

-----

### **4. Python Code to Verify Cost**

Since video encoding (key\_frames) can slightly alter the exact token count, use this script to see the exact price before running a large batch.

```python
import google.generativeai as genai
import time

# Configure your API key
genai.configure(api_key="YOUR_API_KEY")

def get_video_cost(video_path):
    # 1. Upload Video (Free)
    print(f"Uploading {video_path}...")
    video_file = genai.upload_file(path=video_path)
    
    # Wait for processing to complete
    while video_file.state.name == "PROCESSING":
        print("Processing video...")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")

    # 2. Count Tokens (Free) - specific to Gemini 3 Pro
    model = genai.GenerativeModel("gemini-3-pro-preview")
    
    # accurately simulate the request prompt
    response = model.count_tokens([video_file, "Analyze this video."])
    token_count = response.total_tokens
    
    # 3. Calculate Price
    # Pricing: $2.00/1M for <=200k, $4.00/1M for >200k
    if token_count <= 200_000:
        rate = 2.00
        tier = "Standard"
    else:
        rate = 4.00
        tier = "Long Context"

    cost = (token_count / 1_000_000) * rate

    print(f"\n--- Cost Analysis ---")
    print(f"Token Count: {token_count:,}")
    print(f"Pricing Tier: {tier}")
    print(f"Estimated Cost: ${cost:.5f}")

    # Cleanup (Optional)
    # genai.delete_file(video_file.name)

# Usage
# get_video_cost("path_to_your_video.mp4")
```
