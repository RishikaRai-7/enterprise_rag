from dataclasses import dataclass

from app.models.retrieval import RetrievalResult


@dataclass
class CitationResult:
    """
    Represents the verification result
    for a single claim-citation pair.
    """

    claim: str

    citation: int

    supported: bool

    reason: str

    chunk: RetrievalResult