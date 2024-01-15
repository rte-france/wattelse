from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder

from wattelse.chatbot import MAX_TOKENS
from wattelse.chatbot.backend.utils import extract_n_most_relevant_extracts, generate_RAG_prompt
from wattelse.common import TEXT_COLUMN
from wattelse.llm.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION
from wattelse.llm.vars import TEMPERATURE

def initialize_embedding_model(embedding_model_name: str):
    """Load embedding_model"""
    logger.info("Initializing embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    # Fix model max input length issue
    if embedding_model.max_seq_length == 514:
        embedding_model.max_seq_length = 512
    return embedding_model

def initialize_reranker_model(reranker_model_name: str):
    """Load embedding_model and reranker_model"""
    logger.info("Initializing embedding and reranker models...")
    reranker_model = CrossEncoder(reranker_model_name)
    return reranker_model


def enrich_query(llm_api, query: str, history):
    """Use recent interaction context to enrich the user query"""
    enriched_query = llm_api.generate(FR_USER_MULTITURN_QUESTION_SPECIFICATION.format(history=history, query=query),
                                      temperature=TEMPERATURE,
                                      max_tokens=MAX_TOKENS,
                                      )
    return enriched_query


def query_oracle(query: str, data, history=None, **kwargs):
    enriched_query = query
    if kwargs["remember_recent_messages"]:
        enriched_query = enrich_query(kwargs["llm_api"],query, history)
        logger.debug(enriched_query)
    else:
        history = ""


    relevant_extracts, relevant_extracts_similarity = extract_n_most_relevant_extracts(
        kwargs["top_n_extracts"],
        enriched_query,
        data,
        kwargs["docs_embeddings"],
        kwargs["embedding_model"],
        kwargs["bm25_model"],
        retrieval_mode=kwargs["retrieval_mode"],
        reranker_model=kwargs.get("reranker_model"),
        similarity_threshold=kwargs["similarity_threshold"],
    )
    # Generates prompt
    prompt = generate_RAG_prompt(
        query,
        [extract[TEXT_COLUMN] for extract in relevant_extracts],
        expected_answer_size=kwargs["expected_answer_size"],
        custom_prompt=kwargs["custom_prompt"],
        history=history,
    )
    logger.debug(f"Prompt : {prompt}")
    # Generates response
    stream_response = kwargs["llm_api"].generate(prompt,
                                       # system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
                                       temperature=TEMPERATURE,
                                       max_tokens=MAX_TOKENS,
                                       stream=True)
    return relevant_extracts, relevant_extracts_similarity, stream_response
