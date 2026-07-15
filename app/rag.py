from app.generation.generator import Generator
from app.ingestion.indexer import Indexer
from app.models.document import ChunkingStrategy
from app.models.generation import GenerationResult


class EnterpriseRAG:
    """
    High-level interface for the Enterprise RAG system.

    Responsibilities:
    - Document ingestion
    - Question answering

    Internal implementation details remain hidden.
    """

    def __init__(
        self,
        generator: Generator | None = None,
        indexer: Indexer | None = None,
    ):

        self.generator = (
            generator
            if generator is not None
            else Generator()
        )

        self.indexer = (
            indexer
            if indexer is not None
            else Indexer()
        )

    # --------------------------------------------------
    # Ingestion
    # --------------------------------------------------

    def ingest(
        self,
        document_path: str,
        strategy: str = "recursive",
    ):

        strategy_map = {
            "fixed": ChunkingStrategy.FIXED,
            "recursive": ChunkingStrategy.RECURSIVE,
            "semantic": ChunkingStrategy.SEMANTIC,
        }

        chunking_strategy = strategy_map.get(
            strategy.lower(),
            ChunkingStrategy.RECURSIVE,
        )

        self.indexer.index_document(
            file_path=document_path,
            strategy=chunking_strategy,
        )

    def ingest_directory(
        self,
        directory: str,
        strategy: str = "recursive",
    ):

        strategy_map = {
            "fixed": ChunkingStrategy.FIXED,
            "recursive": ChunkingStrategy.RECURSIVE,
            "semantic": ChunkingStrategy.SEMANTIC,
        }

        chunking_strategy = strategy_map.get(
            strategy.lower(),
            ChunkingStrategy.RECURSIVE,
        )

        self.indexer.index_directory(
            directory=directory,
            strategy=chunking_strategy,
        )

    # --------------------------------------------------
    # Question Answering
    # --------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> GenerationResult:

        return self.generator.generate(
            question
        )

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------

    def stats(self):

        return self.indexer.get_stats()