import json

from app.generation.prompt_builder import PromptBuilder
from app.models.citation import CitationResult
from app.models.context import PromptContext
from app.services.llm_service import LLMService, llm_service


class CitationVerifier:
    """
    Verifies every claim against every cited chunk.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        llm_service_instance: LLMService | None = None,
    ):

        self.prompt_builder = (
            prompt_builder
            if prompt_builder
            else PromptBuilder()
        )

        self.llm_service = (
            llm_service_instance
            if llm_service_instance is not None
            else llm_service
        )

    def verify(
        self,
        claims: list[tuple[str, list[int]]],
        context: PromptContext,
    ) -> list[CitationResult]:

        results = []

        for claim, citations in claims:

            for citation in citations:

                if citation not in context.citations:

                    continue

                chunk = context.citations[citation]

                prompt = (
                    self.prompt_builder
                    .build_verification_prompt(
                        claim=claim,
                        evidence=chunk.document,
                    )
                )

                response = self.llm_service.generate(
                    prompt
                )

                try:

                    verdict = json.loads(
                        response.text
                    )

                    supported = verdict.get(
                        "supported",
                        False,
                    )

                    reason = verdict.get(
                        "reason",
                        "",
                    )

                except Exception:

                    supported = False

                    reason = (
                        "Could not parse verifier response."
                    )

                results.append(

                    CitationResult(

                        claim=claim,

                        citation=citation,

                        supported=supported,

                        reason=reason,

                        chunk=chunk,

                    )

                )

        return results