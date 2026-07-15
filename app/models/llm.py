from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Standard response returned by all LLM services.
    """

    text: str

    latency_ms: float

    model: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None