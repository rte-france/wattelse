import json
from typing import Iterable, Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from wattelse.api.openai.client_openai_api import GPT_FAMILY
from wattelse.rag_backend.configs.settings import GeneratorConfig


class RAGError(Exception):
    pass


def get_chat_model(generator_config: GeneratorConfig) -> BaseChatModel:
    endpoint = generator_config.openai_endpoint
    if "azure.com" not in endpoint:
        llm_config = {
            "openai_api_key": generator_config.openai_api_key,
            "openai_api_base": generator_config.openai_endpoint,
            "model_name": generator_config.openai_default_model,
            "temperature": generator_config.temperature,
        }
        return ChatOpenAI(**llm_config)
    else:
        llm_config = {
            "openai_api_key": generator_config.openai_api_key,
            "azure_endpoint": generator_config.openai_endpoint,
            "azure_deployment": generator_config.openai_default_model,
            "api_version": generator_config.azure_api_version,
            "temperature": generator_config.temperature,
        }
        llm = AzureChatOpenAI(**llm_config)
        llm.model_name = generator_config.openai_default_model
        # Workaround to make it work with non OpenAI model on Azure
        if GPT_FAMILY not in llm.model_name:
            llm.model_name = None
            llm.root_client.base_url = generator_config.openai_endpoint
        return llm


def preprocess_streaming_data(streaming_data: Iterable[Any]):
    """Generator to preprocess the streaming data coming from LangChain `rag_chain.stream()`.
    First sent chunk contains relevant_extracts in a convenient format.
    Following chunks contain the actual response from the model token by token.
    """
    for chunk in streaming_data:
        context_chunk = chunk.get("context", None)
        if context_chunk is not None:
            relevant_extracts = [
                {"content": s.page_content, "metadata": s.metadata}
                for s in context_chunk
            ]
            relevant_extracts = {"relevant_extracts": relevant_extracts}
            yield json.dumps(relevant_extracts)
        answer_chunk = chunk.get("answer")
        if answer_chunk:
            yield json.dumps(chunk)


def filter_history(history: list[dict], window_size: int):
    """
    Return last `window_size` interactions (question + answer) from a history.
    """
    return history[-2 * window_size :]


def get_history_as_text(history: list[dict[str, str]]) -> str:
    """
    Format conversation history as a text string.
    """
    history_as_text = ""
    if history is not None:
        for turn in history:
            history_as_text += f"{turn['role']}: {turn['content']}\n"
    return history_as_text
