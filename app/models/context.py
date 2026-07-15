from dataclasses import dataclass
from typing import Dict

from app.models.retrieval import RetrievalResult


@dataclass
class PromptContext:

    formatted_context: str

    formatted_chunks: list[str]

    citations: dict[int, RetrievalResult]