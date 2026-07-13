import time
import uuid
from pathlib import Path
from typing import List

from torch import chunk
from tqdm import tqdm
from langchain_core.documents import Document

from app.database.vector_store import VectorStore
from app.database.chroma import ChromaVectorStore
from app.ingestion.chunker import Chunker
from app.ingestion.embeddings import embedding_service
from app.ingestion.loader import DocumentLoader
from app.models.document import (
    ChunkingStrategy,
    DocumentChunk,
)
from app.retrieval.sparse_index import SparseIndex

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

def __init__(
    self,
    vector_store: VectorStore | None = None,
    sparse_index: RetrievalIndex | None = None,
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

        self.sparse_index.add_documents(

            ids=[
                c.id
                for c in document_chunks
            ],

            documents=texts,

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
    
    def index_directory(
    self,
    directory: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
):
        """
        Index every supported document inside a folder.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"{directory} does not exist."
            )

        files = [

            file

            for file in directory.rglob("*")

            if file.is_file()

        ]

        indexed = 0

        skipped = 0

        total_chunks = 0

        start_time = time.time()

        for file in tqdm(
            files,
            desc="Indexing Documents",
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
                    f"❌ {file.name}: {e}"
                )

        elapsed = time.time() - start_time

        print(" INGESTION SUMMARY")
        print(f"Indexed Files : {indexed}")
        print(f"Skipped Files : {skipped}")
        print(f"Total Chunks  : {total_chunks}")
        print(f"Elapsed Time  : {elapsed:.2f} sec")

    def get_stats(self):
        """
        Return collection statistics.
        """

        return {

            "total_chunks": self.vector_store.count()

        }