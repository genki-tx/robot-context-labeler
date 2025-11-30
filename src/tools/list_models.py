import os
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")

def list_my_models():
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=API_KEY)
    
    print("--- Fetching Available Models ---")
    try:
        # Pager for listing models
        for model in client.models.list():
            # Use getattr to safely access attributes that might be missing in some SDK versions
            methods = getattr(model, 'supported_generation_methods', [])
            
            # If the list is empty (attribute missing), we assume we want to see it just in case,
            # or you can strictly filter: if "generateContent" in methods:
            
            # Simple filter: Only show if we know it supports generateContent OR if we can't tell.
            if not methods or "generateContent" in methods:
                print(f"ID: {model.name}")
                print(f"    Display Name: {getattr(model, 'display_name', 'N/A')}")
                print(f"    Version: {getattr(model, 'version', 'N/A')}")
                print("-" * 30)
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_my_models()