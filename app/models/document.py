from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"

@dataclass
class DocumentChunk:
    """
    Represents one chunk of a document along with all metadata
    required for retrieval, reranking, and citation.
    """

    # Unique identifier
    id: str

    # Actual chunk text
    text: str

    # Original document filename
    source: str

    # Page number (None for txt/markdown)
    page: Optional[int]

    # Section heading if available
    section: Optional[str]

    # Position of this chunk inside the document
    chunk_index: int

    # Which chunking strategy created it
    strategy: ChunkingStrategy

    # Number of characters in the chunk
    char_count: int