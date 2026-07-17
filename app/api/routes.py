from pathlib import Path
import shutil

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.api.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    SourceResponse,
)
from app.rag import EnterpriseRAG

router = APIRouter(
    prefix="/v1",
    tags=["Enterprise RAG"],
)

# --------------------------------------------------
# Upload Directory
# --------------------------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(
    exist_ok=True,
)

# --------------------------------------------------
# Lazy initialization
# --------------------------------------------------

rag: EnterpriseRAG | None = None


def get_rag() -> EnterpriseRAG:
    """
    Lazily create the RAG system.
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

    try:

        result = get_rag().ask(
            request.question
        )

        return AskResponse(
            answer=result.answer,
            confidence=result.confidence,
            model=result.model,
            latency_ms=result.latency_ms,
            retrieved_chunks=len(
                result.retrieved_chunks
            ),
            citations=len(
                result.citation_results
            ),
            sources=[
                SourceResponse(
                    title=chunk.metadata.get(
                        "source",
                        "Unknown Document",
                    ),
                    score=chunk.score,
                )
                for chunk in result.retrieved_chunks
            ],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# --------------------------------------------------
# Upload
# --------------------------------------------------

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):

    try:

        destination = (
            UPLOAD_DIR / file.filename
        )

        with destination.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        chunks = get_rag().ingest(
            document_path=str(destination),
            strategy="recursive",
        )

        return {
            "message": "Document indexed successfully.",
            "filename": file.filename,
            "chunks": (
                len(chunks)
                if chunks is not None
                else None
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# --------------------------------------------------
# Ingest (Local Path)
# --------------------------------------------------

@router.post("/ingest")
async def ingest(
    request: IngestRequest,
):

    try:

        chunks = get_rag().ingest(
            document_path=request.document_path,
            strategy=request.strategy,
        )

        return {
            "message": "Document indexed successfully.",
            "chunks": (
                len(chunks)
                if chunks is not None
                else None
            ),
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

    try:

        documents = []

        if UPLOAD_DIR.exists():

            for file in sorted(
                UPLOAD_DIR.iterdir()
            ):

                if file.is_file():

                    documents.append(
                        {
                            "name": file.name,
                            "size": file.stat().st_size,
                        }
                    )

        return {
            "documents": documents
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )