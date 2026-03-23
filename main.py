from pathlib import Path

from dotenv import load_dotenv
from langchain_core.globals import set_debug

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.graph.graph import wardrobe_graph


def run(user_id: str, user_input: str, file_path: str = None) -> str:
    initial_state = {
        "user_id": user_id,
        "user_input": user_input,
        "intent": None,
        "file_path": file_path,
        "response": None,
        "error": None,
    }
    set_debug(True)
    result = wardrobe_graph.invoke(initial_state)
    return result["response"]


if __name__ == "__main__":
    # # # upload a wardrobe item
    # print(run(
    #     user_id="1",
    #     user_input="upload my new kurta",
    #     file_path=str(BASE_DIR / "sample/wardrobe/red_dress.jpg")
    # ))
    #
    # # upload profile photo
    # print(run(
    #     user_id="1",
    #     user_input="upload my profile photo",
    #     file_path="/Users/kthangaraj/PycharmProjects/wardrobe-ai-hybrid/sample/photo/my_photo.jpg"
    # ))

    # # suggest an outfit
    print(run(
        user_id="1",
        user_input="suggest me an outfit for Diwali",
    ))