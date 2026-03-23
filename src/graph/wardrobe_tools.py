from langchain_core.tools import tool

from src.db.ProfileRepository import ProfileRepository
from src.retriever.HybridRetriever import HybridRetriever

profile_repository = ProfileRepository()
hybrid_retriever = HybridRetriever()


@tool
def get_user_profile(user_id: str) -> str:
    """Fetch the user's style profile including body type, skin tone and preferences."""
    profile = profile_repository.get_profile(user_id)
    if not profile:
        return f"No profile found for user {user_id}. Ask them to upload a profile photo."
    return str(profile)

@tool
def retrieve_wardrobe_items(query: str) -> str:
    """Retrieve relevant wardrobe items from the vector store for a given styling query."""
    items = hybrid_retriever.retrieve_wardrobe(query)
    if not items:
        return "No wardrobe items found. Ask the user to upload clothing items first."
    return "\n".join(f"- {item}" for item in items)