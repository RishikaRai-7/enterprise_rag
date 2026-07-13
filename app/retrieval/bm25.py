from typing import List

from app.models.retrieval import RetrievalResult
from app.retrieval.retrieval_index import RetrievalIndex
from app.retrieval.sparse_index import SparseIndex


class BM25Retriever:

    def __init__(
        self,
        sparse_index: RetrievalIndex | None = None,
        top_k: int = 10,
    ):

        self.top_k = top_k

        self.sparse_index = (

            sparse_index

            if sparse_index is not None

            else SparseIndex()

        )

    def search(
        self,
        query: str,
    ) -> List[RetrievalResult]:

        return self.sparse_index.search(

            query=query,

            top_k=self.top_k,

        )