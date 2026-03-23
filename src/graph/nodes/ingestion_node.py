import os

from src.db.ProfileRepository import ProfileRepository
from src.graph.WardrobeState import WardrobeState
from src.loader.IngestionHandler import IngestionHandler
from src.loader.UserProfileLoader import UserProfileLoader
from src.loader.WardrobeAnalyser import WardrobeAnalyser

model=os.getenv("model")

def analyse_and_store_wardrobe(state: WardrobeState) -> WardrobeState:
    """
        Upload a wardrobe item to the vector store after performing embeddings
    """
    try:
        file_path=state['file_path']
        analyser = WardrobeAnalyser(model)
        wardrobe_details=analyser.analyze(file_path)

        ingestionPipeline=IngestionHandler()
        ingestionPipeline.ingest_wardrobe(wardrobe_details,file_path)
        state['response'] = f"Wardrobe item uploaded successfully from {file_path}"
        return state
    except Exception as e:
        state['error'] = str(e)
        return state




def analyse_and_store_profile(state: WardrobeState) -> WardrobeState:
    """
    Upload a user profile item to the vector store after performing embeddings
    """
    try:
        file_path=state['file_path']
        user_id=state['user_id']
        analyser = UserProfileLoader(model)
        profile_details=analyser.analyze(file_path)

        profile_repository = ProfileRepository()
        profile_repository.upsert_profile(user_id,profile_details)

        state['response'] = f"Profile item uploaded successfully from {file_path}"
        return state
    except Exception as e:
        state['error'] = str(e)
        return state
