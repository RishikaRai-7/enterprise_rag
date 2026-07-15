from pathlib import Path

from app.models.context import PromptContext


class PromptBuilder:
    """
    Loads prompt templates and builds prompts
    for different LLM tasks.
    """

    def __init__(self):

        self.prompt_directory = (
            Path(__file__).parent.parent
            / "prompts"
        )

    def load_prompt(
        self,
        filename: str,
    ) -> str:
        """
        Loads a prompt template from disk.
        """

        return (
            self.prompt_directory / filename
        ).read_text(
            encoding="utf-8"
        )

    def build_generation_prompt(
        self,
        question: str,
        context: PromptContext,
    ) -> str:

        system_prompt = self.load_prompt(
            "system.txt"
        )

        return f"""
SYSTEM
==============================

{system_prompt}

CONTEXT
==============================

{context.formatted_context}

QUESTION
==============================

{question}

ANSWER
==============================
""".strip()

    def build_verification_prompt(
        self,
        claim: str,
        evidence: str,
    ) -> str:

        prompt = self.load_prompt(
            "citation_verification.txt"
        )

        return f"""
{prompt}

Claim:
{claim}

Evidence:
{evidence}
""".strip()

    def build_confidence_prompt(
        self,
        question: str,
        answer: str,
        context: PromptContext,
    ) -> str:

        prompt = self.load_prompt(
            "confidence_scoring.txt"
        )

        return f"""
{prompt}

Question:
{question}

Answer:
{answer}

Context:

{context.formatted_context}
""".strip()

    def build_evaluation_prompt(
        self,
        question: str,
        answer: str,
        context: PromptContext,
    ) -> str:

        prompt = self.load_prompt(
            "answer_evaluation.txt"
        )

        return f"""
{prompt}

Question:
{question}

Answer:
{answer}

Context:

{context.formatted_context}
""".strip()