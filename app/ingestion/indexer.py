import time
import uuid
from pathlib import Path
from typing import List

from torch import chunk
from tqdm import tqdm
from langchain_core.documents import Document

from app.database.chroma import ChromaVectorStore
from app.ingestion.chunker import Chunker
from app.ingestion.embeddings import embedding_service
from app.ingestion.loader import DocumentLoader
from app.models.document import (
    ChunkingStrategy,
    DocumentChunk,
)

class Indexer:
    """
    Coordinates the complete ingestion pipeline.

    Responsibilities:
    - Load documents
    - Chunk documents
    - Convert to DocumentChunk
    - Generate embeddings
    - Store vectors
    """

    def __init__(self):

        self.loader = DocumentLoader()

        self.chunker = Chunker()

        self.vector_store = ChromaVectorStore()

    def _create_document_chunk(
        self,
        chunk: Document,
        index: int,
        strategy: ChunkingStrategy,
    ) -> DocumentChunk:
        metadata = chunk.metadata

        return DocumentChunk(
            id=str(uuid.uuid4()),
            text=chunk.page_content,
            source=Path(
                metadata.get(
                    "source",
                    "",
                )
            ).name,
            page=metadata.get("page"),
            section=metadata.get("section"),
            chunk_index=index,
            strategy=strategy,
            char_count=len(chunk.page_content),
        )

    def _prepare_metadata(
        self,
        chunk: DocumentChunk,
    ) -> dict:
        return {

            "source": chunk.source,

            "page": chunk.page,

            "section": chunk.section,

            "chunk_index": chunk.chunk_index,

            "strategy": chunk.strategy.value,

            "char_count": chunk.char_count,

        }
    
    def index_document(
        self,
        file_path: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ):
        """
        Index a single document.
        """

        start_time = time.time()

        print(f"\n📄 Indexing: {Path(file_path).name}")

        # ------------------------
        # Load
        # ------------------------
        documents = self.loader.load(file_path)

        print(f"Loaded {len(documents)} document(s).")

        # ------------------------
        # Chunk
        # ------------------------
        chunks = self.chunker.chunk(
            documents,
            strategy,
        )

        print(f"Created {len(chunks)} chunks.")

        # ------------------------
        # Convert
        # ------------------------
        document_chunks = [
            self._create_document_chunk(
                chunk,
                index,
                strategy,
            )
            for index, chunk in enumerate(chunks)
        ]

        # ------------------------
        # Embeddings
        # ------------------------
        texts = [chunk.text for chunk in document_chunks]

        embeddings = embedding_service.embed_documents(texts)

        # ------------------------
        # Store
        # ------------------------
        self.vector_store.add_documents(
            ids=[chunk.id for chunk in document_chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self._prepare_metadata(chunk)
                for chunk in document_chunks
            ],
        )

        elapsed = time.time() - start_time

        print(
            f"✅ Indexed {len(document_chunks)} chunks "
            f"in {elapsed:.2f} sec.\n"
        )
        return document_chunks