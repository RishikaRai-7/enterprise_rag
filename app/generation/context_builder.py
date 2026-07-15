from typing import List

from app.models.context import PromptContext
from app.models.retrieval import RetrievalResult


class ContextBuilder:
    """
    Builds the formatted context supplied to the LLM
    from retrieved document chunks.
    """

    def build(
        self,
        results: List[RetrievalResult],
    ) -> PromptContext:

        formatted_chunks = []
        citations = {}

        for index, result in enumerate(results, start=1):

            metadata = result.metadata

            source = metadata.get("source", "Unknown")
            section = metadata.get("section", "N/A")
            page = metadata.get("page", "N/A")

            chunk = (
                f"### Context [{index}]\n\n"
                f"Source : {source}\n"
                f"Section: {section}\n"
                f"Page   : {page}\n\n"
                f"{result.document}"
            )

            formatted_chunks.append(chunk)
            citations[index] = result

        prompt = "\n\n".join(formatted_chunks)

        return PromptContext(
            formatted_context=prompt,
            formatted_chunks=formatted_chunks,
            citations=citations,
        )