import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent

def get_gemini_client():
    try:
        load_dotenv(BASE_DIR / ".env")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_client=genai.Client(api_key=gemini_api_key)
        return gemini_client
    except Exception as e:
        print(f"Exception occured while creating a gemini client: {e}")
        raise
