from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import WeightedReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class RetrievalPipeline:
    """
    Complete retrieval pipeline.

    Pipeline:

        Dense Retrieval
                +
        BM25 Retrieval
                ↓
      Weighted Reciprocal Rank Fusion
                ↓
      Cross Encoder Reranking
                ↓
          Top-K Results
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else HybridRetriever(
                dense_retriever=DenseRetriever(),
                bm25_retriever=BM25Retriever(),
                fusion=WeightedReciprocalRankFusion(),
            )
        )

        self.reranker = (
            reranker
            if reranker is not None
            else CrossEncoderReranker()
        )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        """
        Executes the complete retrieval pipeline.
        """

        # -----------------------------
        # Hybrid Retrieval
        # -----------------------------
        retrieved = self.retriever.search(query)

        # -----------------------------
        # Cross Encoder Reranking
        # -----------------------------
        reranked = self.reranker.rerank(
            query=query,
            results=retrieved,
        )

        return reranked