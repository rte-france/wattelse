from pydantic import field_validator
from pydantic_settings import BaseSettings
import wattelse.rag_backend.prompts as prompts


class RetrieverConfig(BaseSettings):
    """Configuration for the retriever."""

    top_n_extracts: int
    retrieval_method: str
    similarity_threshold: float
    multi_query_mode: bool
    embedding_api_url: str
    embedding_model_name: str


class GeneratorConfig(BaseSettings):
    """Configuration for the generator."""

    openai_api_key: str
    openai_endpoint: str
    openai_default_model: str
    azure_api_version: str | None = None
    remember_recent_messages: bool
    temperature: float
    system_prompt: str
    user_prompt: str
    system_prompt_query_contextualization: str
    user_prompt_query_contextualization: str

    # Custom validator to resolve prompt variables to their actual values
    @field_validator(
        "system_prompt",
        "user_prompt",
        "system_prompt_query_contextualization",
        "user_prompt_query_contextualization",
        mode="before",
    )
    @classmethod
    def resolve_prompt(cls, value: str) -> str:
        if hasattr(prompts, value):
            return getattr(prompts, value)
        raise ValueError(f"Prompt variable {value} not found in prompts.py")


class RAGBackendConfig(BaseSettings):
    """Configuration for the RAG backend."""

    retriever: RetrieverConfig
    generator: GeneratorConfig
