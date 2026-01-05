import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ API Key missing from .env")
    exit()

genai.configure(api_key=api_key)

print("🔍 Scanning available models for your Key...")
found_any = False

try:
    # List all models available to your API key
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   ✅ Available: {m.name}")
            found_any = True

    if not found_any:
        print("⚠️ No chat models found. Check if 'Generative Language API' is enabled in Google Cloud.")

except Exception as e:
    print(f"💥 Error listing models: {e}")