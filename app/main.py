from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Production-grade Retrieval-Augmented "
        "Generation system with hybrid retrieval, "
        "citation verification, and confidence scoring."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
async def root():
    """
    Health check endpoint.
    """

    return {
        "message": "Enterprise RAG API is running."
    }