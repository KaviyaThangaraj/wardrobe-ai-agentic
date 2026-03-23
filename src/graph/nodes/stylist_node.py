import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.graph.WardrobeState import WardrobeState
from src.graph.wardrobe_tools import get_user_profile, retrieve_wardrobe_items

llm = ChatGoogleGenerativeAI(
    model=os.getenv("model"),
    google_api_key=os.getenv("GEMINI_API_KEY")
)

stylist_agent = create_react_agent(
    model=llm,
    tools=[get_user_profile, retrieve_wardrobe_items],
    prompt="""You are a personal fashion stylist specialising in South Asian fashion.
    Always fetch the user profile first, then retrieve relevant wardrobe items.
    Suggest specific outfit combinations that work for the user's body type and skin tone.
    Reference actual wardrobe items and explain why they work together."""
)


def _extract_text(content):
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


def stylist_node(state: WardrobeState) -> WardrobeState:
    try:
        result = stylist_agent.invoke(
            {"messages": [HumanMessage(
                content=f"The user_id is '{state['user_id']}'. Their question: {state['user_input']}"
            )]},
            config={"configurable": {}},
        )
        return {
            **state,
            "response": _extract_text(result["messages"][-1].content)
        }
    except Exception as e:
        return {**state, "error": str(e)}
