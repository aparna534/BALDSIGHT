import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List all available models
models = genai.list_models()

print("✅ Available Gemini Models:")
for model in models:
    print(f"- {model.name} (supports: {model.supported_generation_methods})")

