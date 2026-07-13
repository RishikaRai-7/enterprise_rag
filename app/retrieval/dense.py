from typing import List

from app.database.chroma import ChromaVectorStore
from app.database.vector_store import VectorStore
from app.ingestion.embeddings import embedding_service
from app.models.retrieval import RetrievalResult


class DenseRetriever:
    """
    Performs semantic retrieval using vector embeddings.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int = 10,
    ):

        self.top_k = top_k

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

    def search(
        self,
        query: str,
    ) -> List[RetrievalResult]:

        query_embedding = embedding_service.embed_query(
            query
        )

        return self.vector_store.search(
            embedding=query_embedding,
            top_k=self.top_k,
        )