from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

from wattelse.chatbot import (
    MAX_TOKENS,
    RETRIEVAL_HYBRID_RERANKER,
    RETRIEVAL_BM25,
    RETRIEVAL_HYBRID,
    FASTCHAT_LLM,
    OLLAMA_LLM,
    CHATGPT_LLM,
    TEMPERATURE,
)
from wattelse.chatbot.backend.utils import (
    extract_n_most_relevant_extracts,
    generate_RAG_prompt,
    make_docs_BM25_indexing,
    load_data,
    generate_query_prompt,
)
from wattelse.common import TEXT_COLUMN
from wattelse.api.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION
from wattelse.api.fastchat.client_fastchat_api import FastchatAPI
from wattelse.api.ollama.client_ollama_api import OllamaAPI
from wattelse.api.openai.client_openai_api import OpenAI_API
from wattelse.api.embedding.client_embedding_api import EmbeddingAPI

@lru_cache(maxsize=3)
def initialize_embedding_api():
    """Load EmbeddingAPI"""
    logger.info(f"Initializing EmbeddingAPI")
    embedding_api = EmbeddingAPI()
    return embedding_api


@lru_cache(maxsize=2)
def initialize_reranker_model(reranker_model_name: str):
    """Load reranker_model"""
    logger.info(f"Initializing reranker model: {reranker_model_name}")
    reranker_model = CrossEncoder(reranker_model_name)
    return reranker_model


@lru_cache(maxsize=3)
def initialize_llm_api(llm_api_name: str):
    logger.info(f"Initializing LLM API: {llm_api_name}")
    if llm_api_name == FASTCHAT_LLM:
        return FastchatAPI()
    elif llm_api_name == OLLAMA_LLM:
        return OllamaAPI()
    elif llm_api_name == CHATGPT_LLM:
        return OpenAI_API()
    else:
        logger.error(f"Unknow API name : {llm_api_name}")


def enrich_query(llm_api, query: str, history):
    """Use recent interaction context to enrich the user query"""
    enriched_query = llm_api.generate(
        FR_USER_MULTITURN_QUESTION_SPECIFICATION.format(history=history, query=query),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return enriched_query


class ChatBotError(Exception):
    pass


class ChatbotBackEnd:
    def __init__(self, **kwargs):
        logger.debug("(Re)Initialization of chatbot backend")
        self._use_cache = kwargs.get("use_cache")
        self._llm_api = initialize_llm_api(kwargs.get("llm_api_name"))
        self._embedding_api = initialize_embedding_api()
        self._embedding_model_name = self._embedding_api.get_api_model_name()
        self._retrieval_mode = kwargs.get("retrieval_mode")
        self._reranker_model = (
            initialize_reranker_model(kwargs.get("reranker_model_name"))
            if self._retrieval_mode == RETRIEVAL_HYBRID_RERANKER
            else None
        )
        self.data = None
        self.embeddings = None

    @property
    def llm_api(self):
        return self._llm_api

    @property
    def embedding_api(self):
        return self._embedding_api

    @property
    def reranker_model(self):
        return self._reranker_model

    @property
    def bm25_model(self):
        return self._bm25_model

    def initialize_data(self, data_paths: List[Path]):
        """Initializes data (list of paths, may be limited to a single element if
        only one file is loaded: load the data, recreates embedding if needed or use previously
        embeddings available in cache"""
        data_list = None
        embeddings_array = None
        for data_path in data_paths:
            print(data_path)
            data, embs = load_data(
                data_path,
                self.embedding_api,
                embedding_model_name=self._embedding_model_name,
                use_cache=self._use_cache,
            )
            data_list = (
                data
                if data_list is None
                else pd.concat([data_list, data], axis=0).reset_index(drop=True)
            )
            embeddings_array = (
                embs
                if embeddings_array is None
                else np.concatenate((embeddings_array, embs))
            )
        self.data = data_list
        self.embeddings = embeddings_array
        self._bm25_model = (
            make_docs_BM25_indexing(self.data)
            if self._retrieval_mode
            in (RETRIEVAL_BM25, RETRIEVAL_HYBRID, RETRIEVAL_HYBRID_RERANKER)
            else None
        )

    def query_oracle(self, query: str, history=None, **kwargs):
        if self.data is None:
            msg = "Data not provided!"
            logger.error(msg)
            raise ChatBotError(msg)
        if self.embeddings is None:
            msg = "Embeddings not computed!"
            logger.error(msg)
            raise ChatBotError(msg)

        enriched_query = query
        if kwargs.get("remember_recent_messages"):
            enriched_query = enrich_query(self.llm_api, query, history)
            logger.debug(enriched_query)
        else:
            history = ""

        (
            relevant_extracts,
            relevant_extracts_similarity,
        ) = extract_n_most_relevant_extracts(
            query=enriched_query,
            top_n=kwargs.get("top_n_extracts"),
            data=self.data,
            docs_embeddings=self.embeddings,
            embedding_api=self.embedding_api,
            bm25_model=self.bm25_model,
            retrieval_mode=kwargs.get("retrieval_mode"),
            reranker_model=self.reranker_model,
            similarity_threshold=kwargs.get("similarity_threshold"),
        )

        # Generates prompt
        prompt = generate_RAG_prompt(
            query,
            [extract[TEXT_COLUMN] for extract in relevant_extracts],
            expected_answer_size=kwargs.get("expected_answer_size"),
            custom_prompt=kwargs.get("custom_prompt"),
            history=history,
        )
        logger.debug(f"Prompt : {prompt}")
        # Generates response
        stream_response = self.llm_api.generate(
            prompt,
            # system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        return relevant_extracts, relevant_extracts_similarity, stream_response

    def simple_query(self, query: str, history=None, **kwargs):
        enriched_query = query
        if kwargs.get("remember_recent_messages"):
            enriched_query = enrich_query(self.llm_api, query, history)
            logger.debug(enriched_query)
        else:
            history = ""

        # Generates prompt
        prompt = generate_query_prompt(
            enriched_query,
            custom_prompt=kwargs.get("custom_prompt"),
            history=history,
        )
        logger.debug(f"Prompt : {prompt}")
        # Generates response
        stream_response = self.llm_api.generate(
            prompt,
            # system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
            temperature=TEMPERATURE,
            stream=True,
        )
        return stream_response
