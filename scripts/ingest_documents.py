from pathlib import Path

from app.ingestion.indexer import Indexer
from app.models.document import ChunkingStrategy


def main():
    indexer = Indexer()

    data_dir = Path("data/raw")

    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist.")

    indexer.index_directory(
        directory=str(data_dir),
        strategy=ChunkingStrategy.RECURSIVE,
    )

    stats = indexer.get_stats()

    print(f"Total indexed chunks: {stats['total_chunks']}")


if __name__ == "__main__":
    main()