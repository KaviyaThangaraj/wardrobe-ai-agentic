import json
from pathlib import Path

from google.genai import types
from llama_index.core import Document

from src.gemini.GeminiClient import get_gemini_client


class ImageAnalyser:
    def __init__(self,model):
        self.model = model
        self.MIME_TYPES = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        self.client=get_gemini_client()

    def analyse_image(self,image_path: str, prompt: str) -> Document:
        try:
            ext = Path(image_path).suffix.lower()
            mime_type = self.MIME_TYPES.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                image_data = f.read()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        prompt,
                    ],
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                details=json.loads(raw)
                return Document(
                        text=json.dumps(details),
                        metadata={"source": image_path}
                    )
        except Exception as e:
            print(f"Feature extraction  for the image_path: {image_path} with the exception: {e}")
            raise e




