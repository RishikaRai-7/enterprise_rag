from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.generation.quality_pipeline import QualityPipeline
from app.models.generation import GenerationResult
from app.retrieval.pipeline import RetrievalPipeline
from app.services.llm_service import LLMService, llm_service


class Generator:
    """
    Complete Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        quality_pipeline: QualityPipeline | None = None,
        llm_service_instance: LLMService | None = None,
    ):

        self.retrieval_pipeline = (
            retrieval_pipeline
            if retrieval_pipeline is not None
            else RetrievalPipeline()
        )

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder()
        )

        self.quality_pipeline = (
            quality_pipeline
            if quality_pipeline is not None
            else QualityPipeline()
        )

        self.llm_service = (
            llm_service_instance
            if llm_service_instance is not None
            else llm_service
        )

    def generate(
        self,
        question: str,
    ) -> GenerationResult:
        """
        Executes the complete RAG pipeline.
        """

        # -------------------------
        # Retrieval
        # -------------------------

        retrieved_chunks = (
            self.retrieval_pipeline.retrieve(
                question
            )
        )

        # -------------------------
        # Context
        # -------------------------

        context = self.context_builder.build(
            retrieved_chunks
        )

        # -------------------------
        # Prompt
        # -------------------------

        prompt = (
            self.prompt_builder.build_generation_prompt(
                question=question,
                context=context,
            )
        )

        # -------------------------
        # LLM
        # -------------------------

        llm_response = (
            self.llm_service.generate(
                prompt
            )
        )

        # -------------------------
        # Quality
        # -------------------------

        citation_results, confidence = (
            self.quality_pipeline.evaluate(
                answer=llm_response.text,
                context=context,
                retrieved_chunks=retrieved_chunks,
            )
        )

        # -------------------------
        # Final Result
        # -------------------------

        return GenerationResult(
            answer=llm_response.text,
            retrieved_chunks=retrieved_chunks,
            confidence=confidence,
            citation_results=citation_results,
            model=llm_response.model,
            latency_ms=llm_response.latency_ms,
        )