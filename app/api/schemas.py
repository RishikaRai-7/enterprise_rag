from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request model for asking questions.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="User question."
    )


class AskResponse(BaseModel):
    """
    Response returned by the RAG system.
    """

    answer: str

    confidence: float | None

    model: str | None

    latency_ms: float | None


class IngestRequest(BaseModel):
    """
    Request model for document ingestion.
    """

    document_path: str

    strategy: str = "recursive"