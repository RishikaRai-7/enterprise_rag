from typing import List

from rank_bm25 import BM25Okapi

from app.models.retrieval import RetrievalResult
from app.retrieval.retrieval_index import RetrievalIndex


class SparseIndex(RetrievalIndex):

    def __init__(self):

        self.ids = []
        self.documents = []
        self.metadatas = []

        self.tokenized_documents = []

        self.bm25 = None

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
    ) -> None:

        self.ids.extend(ids)

        self.documents.extend(documents)

        self.metadatas.extend(metadatas)

        self.tokenized_documents.extend(

            [
                doc.lower().split()
                for doc in documents
            ]

        )

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:

        if self.bm25 is None:
            return []

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(

            enumerate(scores),

            key=lambda x: x[1],

            reverse=True,

        )[:top_k]

        results = []

        for index, score in ranked:

            results.append(

                RetrievalResult(

                    id=self.ids[index],

                    document=self.documents[index],

                    metadata=self.metadatas[index],

                    score=float(score),

                    retriever="bm25",

                )

            )

        return results