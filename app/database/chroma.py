from typing import List

import chromadb

from app.config.settings import CHROMA_DB_PATH
from app.database.vector_store import VectorStore
from app.models.retrieval import RetrievalResult


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.
    """

    def __init__(
        self,
        collection_name: str = "enterprise_rag",
    ):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[RetrievalResult]:

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        retrieval_results = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieval_results.append(
                RetrievalResult(
                    id=chunk_id,
                    document=document,
                    metadata=metadata,
                    score=distance,
                    retriever="dense",
                )
            )

        return retrieval_results

    def count(self) -> int:
        return self.collection.count()

    def delete(self) -> None:
        self.client.delete_collection(
            self.collection.name
        )