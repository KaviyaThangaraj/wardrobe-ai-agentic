from langchain_core.documents import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.db.QdrantStore import QdrantStore


class IngestionHandler:
    def __init__(self):
        self.vector_store=QdrantStore()
        self.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-base-en-v1.5"
        )

    def ingestion_pipeline(self,vector_store:QdrantVectorStore)->IngestionPipeline:
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=256,
                    chunk_overlap=20,
                ),
                self.embed_model,
            ],
            vector_store=vector_store,
        )

    def ingest_wardrobe(self,document: Document, doc_id: str):
        vector_store =  self.vector_store.get_vector_store("wardrobe")
        pipeline = self.ingestion_pipeline(vector_store)
        document.id_ = doc_id
        pipeline.run(documents=[document])
        print(f"Ingested wardrobe document with id: {doc_id}")



