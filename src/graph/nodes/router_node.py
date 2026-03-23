import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.graph.WardrobeState import WardrobeState

model=os.getenv("model")

llm = ChatGoogleGenerativeAI(model=model,google_api_key=os.getenv("GEMINI_API_KEY"))

ROUTER_PROMPT = """
You are a routing assistant for a wardrobe styling app.
Classify the user's message into exactly one of these intents:

- upload_wardrobe : user wants to upload a clothing item image
- upload_profile  : user wants to upload their profile photo
- suggest         : user wants outfit suggestions or styling advice

User message: {message}

Reply with ONLY the intent label. Nothing else.
"""

def router_node(state: WardrobeState) -> WardrobeState:
    user_input = state["user_input"]

    response = llm.invoke([
        HumanMessage(content=ROUTER_PROMPT.format(message=user_input))
    ])

    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    intent = content.strip().lower()

    if intent not in ("upload_wardrobe", "upload_profile", "suggest"):
        return {
            **state,
            "intent": None,
            "error": f"Could not classify intent from: {user_input}"
        }

    return {
        **state,
        "intent": intent,
    }