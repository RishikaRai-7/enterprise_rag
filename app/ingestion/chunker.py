from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from app.models.document import ChunkingStrategy


class Chunker:
    """
    Handles different document chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def fixed_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits documents into fixed-size chunks.
        """

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return splitter.split_documents(documents)

    def recursive_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits documents recursively while preserving
        semantic structure as much as possible.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        return splitter.split_documents(documents)

    def chunk(
        self,
        documents: List[Document],
        strategy: ChunkingStrategy,
    ) -> List[Document]:
        """
        Dispatches to the selected chunking strategy.
        """

        if strategy == ChunkingStrategy.FIXED:
            return self.fixed_chunk(documents)

        elif strategy == ChunkingStrategy.RECURSIVE:
            return self.recursive_chunk(documents)

        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )