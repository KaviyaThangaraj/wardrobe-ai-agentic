# Wardrobe AI Hybrid

A personal fashion stylist app specializing in South Asian fashion. It uses AI vision to analyze your wardrobe items and profile photo, then provides personalized outfit suggestions based on your body type, skin tone, and existing wardrobe.

## Architecture

The app is built around a **LangGraph** state machine that routes user requests through different processing pipelines, with a **ReAct agent** powering the styling suggestions.

### Graph Flow

```
                   ┌─────────────────────────┐
                   │       User Request       │
                   └────────────┬─────────────┘
                                │
                                v
                   ┌─────────────────────────┐
                   │     Router Node          │
                   │  (Gemini LLM classifies  │
                   │   intent from message)   │
                   └────────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
              v                 v                  v
   ┌──────────────────┐ ┌──────────────┐ ┌────────────────┐
   │   Ingestion Node │ │ Profile Node │ │  Stylist Node  │
   │                  │ │              │ │                │
   │ Gemini Vision    │ │ Gemini Vision│ │ ReAct Agent    │
   │ analyzes clothing│ │ analyzes user│ │ with tools:    │
   │ image, extracts  │ │ photo for    │ │                │
   │ metadata, embeds │ │ body type,   │ │ get_user_      │
   │ into Qdrant      │ │ skin tone,   │ │   profile()    │
   │                  │ │ stores in    │ │ retrieve_      │
   │                  │ │ SQLite       │ │   wardrobe_    │
   │                  │ │              │ │   items()      │
   └────────┬─────────┘ └──────┬───────┘ └───────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               v
                   ┌─────────────────────────┐
                   │     Response Node        │
                   │  (formats final output   │
                   │   or error message)      │
                   └────────────┬─────────────┘
                                │
                               END
```

### How the Router Works

The router node sends the user's message to Gemini with a classification prompt. Gemini returns one of three intent labels:

| Intent | Trigger | Routes to |
|--------|---------|-----------|
| `upload_wardrobe` | "upload my new kurta" | Ingestion Node |
| `upload_profile` | "upload my profile photo" | Profile Node |
| `suggest` | "suggest an outfit for Diwali" | Stylist Node |

If classification fails, the graph short-circuits to the Response node with an error.

### How the Stylist Agent Works

The stylist node uses a LangGraph **ReAct agent** — an LLM that can reason and act by calling tools in a loop:

```
User: "suggest an outfit for Diwali"
  │
  v
Agent thinks: "I need the user's profile first"
  │
  v
Agent calls: get_user_profile("1")
  → Returns: {skin_tone: "medium", body_type: "pear", best_colors: ["emerald green", ...]}
  │
  v
Agent thinks: "Now I need wardrobe items for Diwali"
  │
  v
Agent calls: retrieve_wardrobe_items("Diwali outfit for pear body shape")
  → Returns: matching items from vector store (hybrid search)
  │
  v
Agent generates: Personalized styling advice combining profile + wardrobe items
```

The agent is instructed to always fetch the profile first, then search the wardrobe, and finally combine both to suggest specific outfit combinations with explanations.

### State

All nodes read from and write to a shared `WardrobeState`:

```python
class WardrobeState(TypedDict):
    user_id: str           # unique user identifier
    user_input: str        # raw user message (used for routing and queries)
    intent: str | None     # classified intent
    file_path: str | None  # path to uploaded image
    response: str | None   # final response
    error: str | None      # error message if any
```

## Data Flow

### Wardrobe Upload
```
Clothing image
  → Gemini Vision extracts structured metadata:
      {type, color, fabric, style, occasion, season, pattern, fit}
  → LlamaIndex chunks text (256 chars, 20 overlap)
  → FastEmbed generates dense vectors (BAAI/bge-base-en-v1.5)
  → Stored in Qdrant with both dense + sparse vectors
```

### Profile Upload
```
User photo
  → Gemini Vision analyzes for styling:
      {skin_tone, body_type, face_shape, best_colors,
       avoid_colors, best_necklines, best_silhouettes}
  → Stored as JSON in SQLite (data/profile.db)
```

### Outfit Suggestion
```
User query → Router classifies as "suggest"
  → Stylist ReAct Agent:
      1. get_user_profile() → reads from SQLite
      2. retrieve_wardrobe_items() → hybrid search in Qdrant
      3. Gemini generates personalized styling advice
  → Response returned to user
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph (state machine + ReAct agent) |
| **LLM** | Google Gemini (routing, vision analysis, styling advice) |
| **Vector Store** | Qdrant Cloud (hybrid search: dense + sparse vectors) |
| **Embeddings** | FastEmbed (`BAAI/bge-base-en-v1.5`, 768 dimensions) |
| **Sparse Search** | SPLADE via Qdrant (BM25-style keyword matching) |
| **Profile Storage** | SQLite |
| **Ingestion Pipeline** | LlamaIndex (chunking + embedding) |
| **API** | FastAPI + Uvicorn |
| **Tracing** | LangSmith |

## Project Structure

```
wardrobe-ai-hybrid/
├── app.py                        # FastAPI endpoints
├── main.py                       # CLI entry point
├── pyproject.toml                # Dependencies (managed by uv)
├── .env                          # API keys and config
├── data/
│   └── profile.db                # SQLite user profiles (auto-created)
├── sample/
│   ├── wardrobe/                 # Sample clothing images
│   └── photo/                    # Sample profile photos
└── src/
    ├── db/
    │   ├── ProfileRepository.py  # SQLite CRUD for user profiles
    │   ├── QdrantStore.py        # Qdrant collection setup (dense + sparse)
    │   └── schema.sql            # SQLite schema definition
    ├── gemini/
    │   └── GeminiClient.py       # Google Gemini API client factory
    ├── graph/
    │   ├── WardrobeState.py      # Shared state definition (TypedDict)
    │   ├── graph.py              # Graph construction, routing, compilation
    │   ├── wardrobe_tools.py     # LangChain tools for the stylist agent
    │   └── nodes/
    │       ├── router_node.py    # Intent classification via Gemini
    │       ├── ingestion_node.py # Wardrobe & profile upload handlers
    │       └── stylist_node.py   # ReAct agent for outfit suggestions
    ├── loader/
    │   ├── ImageAnalyser.py      # Gemini Vision wrapper (image → JSON)
    │   ├── IngestionHandler.py   # LlamaIndex embedding + vector store pipeline
    │   ├── UserProfileLoader.py  # Profile photo → styling profile JSON
    │   └── WardrobeAnalyser.py   # Clothing image → metadata JSON
    └── retriever/
        └── HybridRetriever.py    # Qdrant hybrid search (dense + sparse)
```

## Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- A [Gemini API key](https://aistudio.google.com/apikeys)
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster

### Install

```bash
uv sync
```

### Configure

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
model=gemini-2.0-flash

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=warddrobe-ai
```

### Run

**API server:**
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**CLI:**
```bash
uv run python main.py
```

## API Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET` | `/health` | - | Health check |
| `POST` | `/upload/wardrobe` | `{user_id, user_input, file_path}` | Analyze and store a clothing item |
| `POST` | `/upload/profile` | `{user_id, user_input, file_path}` | Analyze and store a user profile photo |
| `POST` | `/suggest` | `{user_id, user_input}` | Get personalized outfit suggestions |

Interactive API docs at `http://localhost:8000/docs`.

### Example Requests

**Upload a wardrobe item:**
```bash
curl -X POST http://localhost:8000/upload/wardrobe \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1", "user_input": "upload my new kurta", "file_path": "/path/to/kurta.jpg"}'
```

**Upload a profile photo:**
```bash
curl -X POST http://localhost:8000/upload/profile \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1", "user_input": "upload my profile photo", "file_path": "/path/to/photo.jpg"}'
```

**Get outfit suggestions:**
```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1", "user_input": "suggest me an outfit for Diwali"}'
```

## Stylist Agent Tools

The ReAct agent has two tools it calls autonomously:

| Tool | Input | Source | Purpose |
|------|-------|--------|---------|
| `get_user_profile` | `user_id` | SQLite | Fetch body type, skin tone, color preferences |
| `retrieve_wardrobe_items` | `query` | Qdrant (hybrid search) | Find matching wardrobe items |

## Vector Store Details

Qdrant is configured with **hybrid search** combining two retrieval strategies:

- **Dense vectors** (`BAAI/bge-small-en-v1.5`, 768 dimensions, cosine similarity) — semantic matching. "Diwali outfit" finds items tagged with "festive" occasions.
- **Sparse vectors** (SPLADE) — keyword/BM25 matching. "kurta" finds items with that exact type.

The hybrid approach ensures relevant results whether the user asks with specific terms ("red silk saree") or general descriptions ("something festive and elegant").
