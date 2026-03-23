from typing import TypedDict


class WardrobeState(TypedDict):
    user_id: str
    user_input: str
    intent: str | None
    file_path: str | None
    response: str | None
    error: str | None
