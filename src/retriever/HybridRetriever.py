from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever

from src.db.QdrantStore import QdrantStore


class HybridRetriever:
    def __init__(self):
        self.vector_store = QdrantStore();
        self.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-base-en-v1.5"
        )
        Settings.embed_model = self.embed_model
        self.similarity_top_k=5

    def get_db_index(self, collection: str)-> VectorStoreIndex:
        vector_store=self.vector_store.get_vector_store(collection)
        return VectorStoreIndex.from_vector_store(vector_store)

    def retrieve_wardrobe(self,query)-> list:
        index=self.get_db_index("wardrobe")
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.similarity_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID
        )
        nodes = retriever.retrieve(query)
        return [n.text for n in nodes]