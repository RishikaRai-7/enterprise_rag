from typing import List, Tuple

from sentence_transformers import CrossEncoder


class CrossEncoderService:
    """
    Singleton service responsible for loading and
    scoring sentence pairs using a Cross Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print(f"Loading Cross Encoder: {model_name}")

        self.model = CrossEncoder(model_name)

        print("Cross Encoder loaded successfully.")

    def score(
        self,
        sentence_pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """
        Returns relevance scores for (query, document) pairs.
        """

        scores = self.model.predict(sentence_pairs)

        return [float(score) for score in scores]


cross_encoder_service = CrossEncoderService()