from parser import load_all_documents
from embedding_pipeline import create_chunks, EmbeddingModel
from langchain_chroma import Chroma
import os
import shutil


class VectorDatabase:
    def __init__(self, db_path: str = r"D:\DRDO PROJECT\GenAI\ChromaDB"):
        self.db_path = db_path
        self.vector_store = None
        self._embedding_model = None

    def load_embedding_model(self):
        embedding = EmbeddingModel()
        embedding.load_model()
        self._embedding_model = embedding.get_embedding_model()
        return self._embedding_model

    def create_vector_store(self, chunks=None, embedding_model=None):
        """
        Build ChromaDB from chunks.
        If chunks is None, loads and chunks documents automatically.
        If embedding_model is None, loads it automatically.
        """
        if embedding_model is None:
            embedding_model = (
                self._embedding_model or self.load_embedding_model()
            )

        if chunks is None:
            docs   = load_all_documents()
            chunks = create_chunks(docs)

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=self.db_path
        )
        return self.vector_store

    def load_vector_store(self, embedding_model=None):
        """
        Open an already-persisted ChromaDB without rebuilding it.
        Use this on startup when the DB already exists.
        """
        if embedding_model is None:
            embedding_model = (
                self._embedding_model or self.load_embedding_model()
            )

        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=embedding_model
        )
        return self.vector_store

    def get_vector_store(self):
        return self.vector_store