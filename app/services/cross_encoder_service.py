from typing import List

from sentence_transformers import CrossEncoder

from app.config.settings import CROSS_ENCODER_MODEL


class CrossEncoderService:
    """
    Service responsible for reranking retrieved documents.

    The model is loaded lazily on the first request.
    """

    def __init__(self):
        self.model = None

    def _get_model(self) -> CrossEncoder:
        """
        Lazily load the Cross Encoder model.
        """

        if self.model is None:

            print(
                f"Loading Cross Encoder model: {CROSS_ENCODER_MODEL}"
            )

            self.model = CrossEncoder(
                CROSS_ENCODER_MODEL
            )

            print(
                "Cross Encoder model loaded successfully."
            )

        return self.model

    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Computes relevance scores for a query-document pair.
        """

        model = self._get_model()

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = model.predict(pairs)

        return scores.tolist()


cross_encoder_service = CrossEncoderService()