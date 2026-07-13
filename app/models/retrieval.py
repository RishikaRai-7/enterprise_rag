from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """
    Represents a single retrieved chunk.
    """

    id: str

    document: str

    metadata: Dict[str, Any]

    score: float

    retriever: str

    rerank_score: float | None = None