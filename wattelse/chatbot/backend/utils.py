import json
from pathlib import Path
from typing import Iterable, Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI

AZURE_API_VERSION = "2024-02-01"

CONFIG_NAME_TO_CONFIG_PATH = {
    "local": Path("wattelse/chatbot/backend/configs/local_rag_config.toml"),
    "azure": Path("wattelse/chatbot/backend/configs/azure_rag_config.toml"),
}


class RAGError(Exception):
    pass


def get_chat_model(generator_config: dict) -> BaseChatModel:
    endpoint = generator_config["openai_endpoint"]
    if "azure.com" not in endpoint:
        llm_config = {
            "openai_api_key": generator_config["openai_api_key"],
            "openai_api_base": generator_config["openai_endpoint"],
            "model_name": generator_config["openai_default_model"],
            "temperature": generator_config["temperature"],
        }
        return ChatOpenAI(**llm_config)
    else:
        llm_config = {
            "openai_api_key": generator_config["openai_api_key"],
            "azure_endpoint": generator_config["openai_endpoint"],
            "azure_deployment": generator_config["openai_default_model"],
            "api_version": AZURE_API_VERSION,
            "temperature": generator_config["temperature"],
        }
        llm = AzureChatOpenAI(**llm_config)
        llm.model_name = generator_config[
            "openai_default_model"
        ]  # with Azure, set to gpt3.5-turbo by default
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
