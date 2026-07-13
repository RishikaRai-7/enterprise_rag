from dataclasses import dataclass
from typing import Dict

from app.models.retrieval import RetrievalResult


@dataclass
class PromptContext:

    prompt: str

    formatted_chunks: list[str]

    citations: dict[int, RetrievalResult]