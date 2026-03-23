import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class QdrantStore:
    COLLECTIONS = {
        "wardrobe": "wardrobe-collection"
    }
    DENSE_DIM = 768  # BAAI/bge-small-en-v1.5 dimension

    def __init__(self):
        load_dotenv(BASE_DIR / ".env")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._ensure_collections()

    def _ensure_collections(self):
        for collection_name in self.COLLECTIONS.values():
            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense": VectorParams(
                            size=self.DENSE_DIM,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    },
                )
                print(f"Created collection: {collection_name}")
            else:
                print(f"Collection already exists: {collection_name}")

    def get_vector_store(self, collection: str) -> QdrantVectorStore:
        collection_name = self.COLLECTIONS[collection]
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            enable_hybrid=True,
            dense_vector_name="text-dense",
            sparse_vector_name="text-sparse",
        )