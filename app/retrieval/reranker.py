from typing import List

from app.models.retrieval import RetrievalResult
from app.services.cross_encoder_service import (
    cross_encoder_service,
)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a Cross Encoder.
    """

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:

        if not results:
            return []

        sentence_pairs = [

            (query, result.document)

            for result in results

        ]

        scores = cross_encoder_service.score(
            sentence_pairs
        )

        reranked_results = []

        for result, score in zip(results, scores):

            reranked_results.append(

                RetrievalResult(

                    id=result.id,

                    document=result.document,

                    metadata=result.metadata,

                    score=result.score,

                    retriever=result.retriever,

                    rerank_score=score,

                )

            )

        reranked_results.sort(

            key=lambda x: x.rerank_score,

            reverse=True,

        )

        return reranked_results[: self.top_k]