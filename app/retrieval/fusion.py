from collections import defaultdict
from typing import List

from app.models.retrieval import RetrievalResult


class WeightedReciprocalRankFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        *retrieval_lists: List[RetrievalResult],
    ) -> List[RetrievalResult]:

        scores = defaultdict(float)

        chunk_lookup = {}

        for results in retrieval_lists:

            for rank, chunk in enumerate(results, start=1):

                scores[chunk.id] += 1 / (
                    self.k + rank
                )

                chunk_lookup[
                    chunk.id
                ] = chunk

        ranked = sorted(

            scores.items(),

            key=lambda x: x[1],

            reverse=True,

        )

        fused = []

        for chunk_id, score in ranked:

            chunk = chunk_lookup[chunk_id]

            chunk.score = score

            chunk.retriever = "hybrid"

            fused.append(chunk)

        return fused
    
