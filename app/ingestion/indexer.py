import time
import uuid
from pathlib import Path

from tqdm import tqdm
from langchain_core.documents import Document

from app.database.chroma import ChromaVectorStore
from app.database.interfaces import VectorStore
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import Chunker
from app.models.document import (
    ChunkingStrategy,
    DocumentChunk,
)
from app.retrieval.sparse_index import SparseIndex
from app.services.embedding_service import embedding_service


class Indexer:
    """
    Coordinates the complete document ingestion pipeline.

    Pipeline:
        Load
        -> Chunk
        -> Embed
        -> Dense Index
        -> Sparse Index
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sparse_index: SparseIndex | None = None,
    ):

        self.loader = DocumentLoader()
        self.chunker = Chunker()

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

        self.sparse_index = (
            sparse_index
            if sparse_index is not None
            else SparseIndex()
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

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
                metadata.get("source", "")
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

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def index_document(
        self,
        file_path: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ) -> list[DocumentChunk]:

        start = time.time()

        documents = self.loader.load(file_path)

        chunks = self.chunker.chunk(
            documents,
            strategy,
        )

        document_chunks = [

            self._create_document_chunk(
                chunk,
                index,
                strategy,
            )

            for index, chunk
            in enumerate(chunks)

        ]

        texts = [
            chunk.text
            for chunk in document_chunks
        ]

        embeddings = embedding_service.embed_documents(
            texts
        )

        metadata = [
            self._prepare_metadata(chunk)
            for chunk in document_chunks
        ]

        self.vector_store.add_documents(
            ids=[
                chunk.id
                for chunk in document_chunks
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
        )

        self.sparse_index.add_documents(
            ids=[
                chunk.id
                for chunk in document_chunks
            ],
            documents=texts,
            metadatas=metadata,
        )

        elapsed = time.time() - start

        print(
            f"Indexed "
            f"{len(document_chunks)} chunks "
            f"in {elapsed:.2f}s"
        )

        return document_chunks

    def index_directory(
        self,
        directory: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ):

        directory = Path(directory)

        if not directory.exists():

            raise FileNotFoundError(
                directory
            )

        indexed = 0
        skipped = 0
        total_chunks = 0

        start = time.time()

        files = [
            file
            for file in directory.rglob("*")
            if file.is_file()
        ]

        for file in tqdm(
            files,
            desc="Indexing",
        ):

            try:

                chunks = self.index_document(
                    str(file),
                    strategy,
                )

                indexed += 1
                total_chunks += len(chunks)

            except ValueError:

                skipped += 1

            except Exception as e:

                skipped += 1
                print(
                    f"{file.name}: {e}"
                )

        elapsed = time.time() - start

        print("\n========== SUMMARY ==========")
        print(f"Indexed Files : {indexed}")
        print(f"Skipped Files : {skipped}")
        print(f"Total Chunks  : {total_chunks}")
        print(f"Elapsed Time  : {elapsed:.2f}s")

    def get_stats(self):

        return {
            "total_chunks": self.vector_store.count()
        }