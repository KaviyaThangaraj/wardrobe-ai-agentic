from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.graph.graph import wardrobe_graph


# --- request models ---

class UploadRequest(BaseModel):
    user_id: str
    user_input: str
    file_path: str

class SuggestRequest(BaseModel):
    user_id: str
    user_input: str


# --- app ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warddrobe AI service starting...")
    print("Graph compiled and ready")
    yield
    print("Warddrobe AI service shutting down...")

app = FastAPI(
    title="Warddrobe AI",
    description="Personal South Asian fashion stylist",
    lifespan=lifespan
)


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload/wardrobe")
def upload_wardrobe(req: UploadRequest):
    try:
        result = wardrobe_graph.invoke({
            "user_id": req.user_id,
            "user_input": req.user_input,
            "intent": None,
            "file_path": req.file_path,
            "response": None,
            "error": None,
        })
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {"response": result["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/profile")
def upload_profile(req: UploadRequest):
    try:
        result = wardrobe_graph.invoke({
            "user_id": req.user_id,
            "user_input": req.user_input,
            "intent": None,
            "file_path": req.file_path,
            "response": None,
            "error": None,
        })
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {"response": result["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggest")
def suggest(req: SuggestRequest):
    try:
        result = wardrobe_graph.invoke({
            "user_id": req.user_id,
            "user_input": req.user_input,
            "intent": None,
            "file_path": None,
            "response": None,
            "error": None,
        })
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {"response": result["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))