from typing import List

from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import WeightedReciprocalRankFusion


class HybridRetriever:
    """
    Combines dense and sparse retrieval
    using Weighted Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        fusion: WeightedReciprocalRankFusion | None = None,
    ):

        self.dense = (
            dense_retriever
            if dense_retriever
            else DenseRetriever()
        )

        self.bm25 = (
            bm25_retriever
            if bm25_retriever
            else BM25Retriever()
        )

        self.fusion = (
            fusion
            if fusion
            else WeightedReciprocalRankFusion()
        )

    def search(
        self,
        query: str,
    ) -> List[RetrievalResult]:

        dense_results = self.dense.search(query)

        bm25_results = self.bm25.search(query)

        return self.fusion.fuse(
            dense_results,
            bm25_results,
        )
