import time

from openai import OpenAI

from app.config.settings import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
)

from app.models.llm import LLMResponse


class LLMService:
    """
    Provider-agnostic LLM service using OpenRouter.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ):

        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        print("Using OpenRouter model:", LLM_MODEL)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        self.model = LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:

        last_exception = None

        for attempt in range(self.max_retries):

            try:

                start = time.perf_counter()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                text = (
                    response.choices[0]
                    .message.content
                    .strip()
                )

                return LLMResponse(
                    text=text,
                    latency_ms=latency_ms,
                    model=self.model,
                )

            except Exception as e:

                print(
                    f"Attempt {attempt + 1} failed:",
                    repr(e),
                )

                last_exception = e

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM request failed after "
            f"{self.max_retries} attempts."
        ) from last_exception


llm_service = LLMService()