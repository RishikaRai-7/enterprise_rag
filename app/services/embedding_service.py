from typing import List

from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL


class EmbeddingService:
    """
    Service responsible for generating embeddings.

    The model is loaded lazily on the first request.
    """

    def __init__(self):
        self.model = None

    def _get_model(self) -> SentenceTransformer:

        if self.model is None:

            print(
                f"Loading embedding model: {EMBEDDING_MODEL}"
            )

            self.model = SentenceTransformer(
                EMBEDDING_MODEL
            )

            print(
                "Embedding model loaded successfully."
            )

        return self.model

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        model = self._get_model()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> List[float]:

        model = self._get_model()

        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()


embedding_service = EmbeddingService()