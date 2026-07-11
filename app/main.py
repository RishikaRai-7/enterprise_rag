from fastapi import FastAPI

app = FastAPI(
    title="Enterprise RAG API",
    description="Hybrid Retrieval-Augmented Generation System",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Enterprise RAG API is running!"
    }