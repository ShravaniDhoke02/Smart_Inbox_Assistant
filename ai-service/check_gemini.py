import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

try:
    import google.generativeai as genai
except Exception as e:
    print("IMPORT_ERROR:", str(e))
    raise SystemExit(1)

api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
model_name = (os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro-latest").strip()
print("API_KEY_SET:", bool(api_key))
print("MODEL:", model_name)

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
    raise SystemExit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Reply with exactly: GEMINI_OK")
    text = response.text.strip() if getattr(response, "text", None) else ""
    print("API_RESPONSE:", text)
    if "GEMINI_OK" in text:
        print("RESULT: Gemini API is working.")
    else:
        print("RESULT: API responded but content did not match the expected test response.")
        raise SystemExit(2)
except Exception as e:
    print("ERROR: Gemini API call failed.")
    print(type(e).__name__, str(e))
    raise SystemExit(3)
