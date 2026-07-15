from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
)
from app.rag import EnterpriseRAG

router = APIRouter(
    prefix="/v1",
    tags=["Enterprise RAG"],
)

# --------------------------------------------------
# Lazy initialization
# --------------------------------------------------

rag: EnterpriseRAG | None = None


def get_rag() -> EnterpriseRAG:
    """
    Lazily create the RAG system.

    This avoids constructing the entire dependency
    graph during FastAPI startup.
    """

    global rag

    if rag is None:
        rag = EnterpriseRAG()

    return rag


# --------------------------------------------------
# Ask
# --------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(request: AskRequest):
    """
    Ask a question to the RAG system.
    """

    try:

        result = get_rag().ask(
            request.question
        )

        return AskResponse(
            answer=result.answer,
            confidence=result.confidence,
            model=result.model,
            latency_ms=result.latency_ms,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# --------------------------------------------------
# Ingest
# --------------------------------------------------

@router.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Ingest a document.
    """

    try:

        get_rag().ingest(
            document_path=request.document_path,
            strategy=request.strategy,
        )

        return {
            "message": "Document indexed successfully."
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# --------------------------------------------------
# Documents
# --------------------------------------------------

@router.get("/documents")
async def documents():
    """
    List indexed documents.
    """

    try:

        return {
            "documents": []
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )