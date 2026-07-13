from abc import ABC, abstractmethod
from typing import List

from app.models.retrieval import RetrievalResult


class VectorStore(ABC):
    """
    Abstract interface for all vector databases.
    """

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass