import time

from google import genai
from google.genai import types

from app.config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from app.models.llm import LLMResponse


class GeminiService:
    """
    Singleton service responsible for interacting
    with Google's Gemini API.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
    ):

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
        )

        self.model = GEMINI_MODEL

        self.config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Generates a response from Gemini.

        Returns:
            LLMResponse
        """

        last_exception = None

        for attempt in range(self.max_retries):

            try:

                start = time.perf_counter()

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.config,
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                text = (
                    response.text.strip()
                    if response.text
                    else ""
                )

                return LLMResponse(
                    text=text,
                    latency_ms=latency_ms,
                    model=self.model,
                )

            except Exception as e:

                last_exception = e

                if attempt < self.max_retries - 1:

                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Gemini request failed after "
            f"{self.max_retries} attempts."
        ) from last_exception


gemini_service = GeminiService()