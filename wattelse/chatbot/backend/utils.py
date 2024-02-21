from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import string

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import Stemmer
from rank_bm25 import BM25Plus

from wattelse.chatbot import (
    CACHE_DIR,
    RETRIEVAL_DENSE,
    RETRIEVAL_BM25,
    RETRIEVAL_HYBRID,
)
from wattelse.common import TEXT_COLUMN
from wattelse.common.cache_utils import save_embeddings, load_embeddings
from wattelse.api.prompts import FR_USER_BASE_RAG, FR_USER_BASE_QUERY
from wattelse.api.embedding.client_embedding_api import EmbeddingAPI

STEMMER = Stemmer.Stemmer("french")


def bm25_preprocessing(text: str):
    """Text preprocessing function before sending it to BM25"""
    text = text.lower()  # lowercase
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # remove punctuation
    tokenized_text = text.split(" ")  # 1 token = 1 word
    tokenized_text = list(filter(None, tokenized_text))  # remove empty strings
    tokenized_text = STEMMER.stemWords(tokenized_text)  # lemmatize tokens
    return tokenized_text


def make_docs_BM25_indexing(data: pd.DataFrame):
    """Index a list of docs using BM25+ algorithm"""
    tokenized_docs = [bm25_preprocessing(doc) for doc in data[TEXT_COLUMN]]
    bm25 = BM25Plus(tokenized_docs)
    return bm25


def compute_bm25_score(query: str, bm25_model: BM25Plus):
    """Computes BM25 score according to a query"""
    query = bm25_preprocessing(query)
    return bm25_model.get_scores(query)


def make_docs_embedding(docs: str | List[str], embedding_api: EmbeddingAPI):
    """Embeds a list of docs using a SentenceTransformer model"""
    return embedding_api.encode(docs)


def compute_dense_embeddings_score(
    query: str, docs_embeddings: np.ndarray, embedding_api: EmbeddingAPI
):
    """Computes similarity score between a query and docs dense embeddings"""
    query_embedding = embedding_api.encode(query)
    return cosine_similarity(query_embedding, docs_embeddings)[0]


def rerank_extracts(
    query: str,
    extracts: List[Dict],
    top_n: int,
    reranker_model: CrossEncoder,
    similarity_threshold: float = 0,
):
    """Rerank extract using a CrossEncoder model. Used after an hybrid search over documents (mode hybrid+reranker).

    Args:
        query (str): query
        extracts (List[Dict]): corpus of documents
        top_n (int): top n extracts to return
        reranker_model (CrossEncoder): reranker model
        similarity_threshold (float, optional): extracts with similarity score lower
                                                than similarity_threshold will be discarded

    Returns:
        Tuple[List[str], List[float]]: list of relevant extracts and their associated similarity score
    """
    docs = [e[TEXT_COLUMN] for e in extracts]
    pairs = [(query, doc) for doc in docs]
    similarity_score = reranker_model.predict(pairs)
    above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
    top_n_indices = above_threshold_indices[
        np.argsort(similarity_score[above_threshold_indices])
    ][::-1][:top_n]
    relevant_extracts = [extracts[i] for i in top_n_indices]
    relevant_extracts_scores = similarity_score[top_n_indices].tolist()
    return relevant_extracts, relevant_extracts_scores


def extract_n_most_relevant_extracts(
    top_n: int,
    query: str,
    data: pd.DataFrame,
    docs_embeddings: np.ndarray,
    embedding_api: EmbeddingAPI,
    bm25_model: BM25Plus,
    retrieval_mode: str = RETRIEVAL_DENSE,
    reranker_model: CrossEncoder = None,
    reranker_ratio: int = 5,
    similarity_threshold: float = 0,
):
    """Select top _n most relevant extarcts according to a user query. 4 retrieval modes are available.

    Args:
        top_n (int): top n extracts to return
        query (str): query
        docs (pd.DataFrame): corpus of documents
        docs_embeddings (np.ndarray): array of document embeddings
        embedding_api (EmbeddingAPI): embedding api
        bm25_model (BM25Plus): BM25 model
        retrieval_mode (str, optional): algorithm to retrieve top_n extracts :
                                        - 'bm25' -> uses BM25 model
                                        - 'dense' -> uses embedding_api
                                        - 'hybrid' -> uses both bm25 and embedding_api
                                        - 'hybrid+reranker' -> uses uses both bm25 and embedding_api and rerank using crossencoder
        reranker_model (CrossEncoder, optional): reranker model
        reranker_ratio (int, optional): reranker_ratio*top_n extracts are send to the reranker model
        similarity_threshold (float, optional): extracts with similarity score lower
                                                than similarity_threshold will be discarded

    Returns:
        Tuple[List[str], List[float]]: list of relevant extracts and their associated similarity score
    """
    extracts = data.reset_index(drop=True).to_dict("records")

    # BM25 model
    if retrieval_mode == RETRIEVAL_BM25:
        similarity_score = compute_bm25_score(query, bm25_model)
        above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
        top_n_indices = above_threshold_indices[
            np.argsort(similarity_score[above_threshold_indices])
        ][::-1][:top_n]
        relevant_extracts = [extracts[i] for i in top_n_indices]
        relevant_extracts_similarity = similarity_score[top_n_indices].tolist()

    # Dense model
    elif retrieval_mode == RETRIEVAL_DENSE:
        similarity_score = compute_dense_embeddings_score(
            query, docs_embeddings, embedding_api
        )
        above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
        top_n_indices = above_threshold_indices[
            np.argsort(similarity_score[above_threshold_indices])
        ][::-1][:top_n]
        relevant_extracts = [extracts[i] for i in top_n_indices]
        relevant_extracts_similarity = similarity_score[top_n_indices].tolist()

    # Hybrid
    elif retrieval_mode == RETRIEVAL_HYBRID:
        dense_similarity_score = compute_dense_embeddings_score(
            query, docs_embeddings, embedding_api
        )
        bm25_similarity_score = compute_bm25_score(query, bm25_model)
        sorted_dense_indices = dense_similarity_score.argsort()[::-1]
        sorted_bm25_indices = bm25_similarity_score.argsort()[::-1]
        logger.debug(
            f"Nombre d'extraits communs : {len(np.intersect1d(sorted_dense_indices[:top_n], sorted_bm25_indices[:top_n]))}"
        )
        # Interleave dense and bm25 top indices to get half from each algorithm
        assert len(sorted_bm25_indices) == len(sorted_dense_indices)  # sanity check
        interleaved_indices = np.vstack(
            (sorted_dense_indices, sorted_bm25_indices)
        ).reshape((-1,), order="F")
        # Remove duplicates
        interleaved_indices = pd.unique(interleaved_indices)
        top_n_indices = interleaved_indices[:top_n]
        relevant_extracts = [extracts[i] for i in top_n_indices]
        relevant_extracts_similarity = np.zeros(
            len(relevant_extracts)
        ).tolist()  # FIXME: get scores for each extract

    # Hybrid + Reranker
    else:
        dense_similarity_score = compute_dense_embeddings_score(
            query, docs_embeddings, embedding_api
        )
        bm25_similarity_score = compute_bm25_score(query, bm25_model)
        top_n_dense_indices = dense_similarity_score.argsort()[::-1][
            : top_n * reranker_ratio
        ]
        top_n_bm25_indices = bm25_similarity_score.argsort()[::-1][
            : top_n * reranker_ratio
        ]
        relevant_extracts = [extracts[i] for i in top_n_dense_indices]
        relevant_extracts_bm25 = [extracts[i] for i in top_n_bm25_indices]
        # merge dense and bm25
        for e in relevant_extracts_bm25:
            if e not in relevant_extracts:
                relevant_extracts.append(e)
        relevant_extracts, relevant_extracts_similarity = rerank_extracts(
            query,
            relevant_extracts,
            top_n,
            reranker_model,
            similarity_threshold=similarity_threshold,
        )

    return relevant_extracts, relevant_extracts_similarity


def generate_RAG_prompt(
    query: str,
    context_elements: List[str],
    expected_answer_size="short",
    custom_prompt=None,
    history=None,
) -> str:
    """
    Generates RAG prompt using query and context.
    """
    context = "\n".join(context_elements)
    expected_answer_size = "courte" if expected_answer_size == "short" else "détaillée"
    if custom_prompt:
        return custom_prompt.format(
            context=context,
            query=query,
            expected_answer_size=expected_answer_size,
            history=history,
        )
    else:
        return FR_USER_BASE_RAG.format(
            context=context, query=query, expected_answer_size=expected_answer_size
        )


def generate_query_prompt(
    query: str,
    custom_prompt=None,
    history=None,
) -> str:
    """
    Generates RAG prompt using query and context.
    """
    if custom_prompt:
        return custom_prompt.format(
            query=query,
            history=history,
        )
    else:
        return FR_USER_BASE_QUERY.format(
            query=query
        )


@lru_cache(maxsize=5)
def load_data(
    data_file: Path,
    embedding_api: EmbeddingAPI,
    embedding_model_name: str = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Loads data and transform them into embeddings; data file shall contain a column 'processed_text' (preferred)
    or 'text'"""
    logger.info(f"Using data from: {data_file}")
    data = pd.read_csv(data_file, keep_default_na=False)
    if TEXT_COLUMN in data.columns:
        docs = data[TEXT_COLUMN]
    else:
        logger.error(
            f"Data {data_file} not formatted correctly! (expecting {TEXT_COLUMN} column"
        )
        sys.exit(-1)

    cache_path = CACHE_DIR / f"{embedding_model_name}_{data_file.name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache: {use_cache}")
    if not use_cache or not cache_path.exists():
        logger.info("Computing sentence embeddings...")
        docs_embeddings = make_docs_embedding(docs.to_list(), embedding_api)
        if use_cache:
            save_embeddings(docs_embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")
    else:
        docs_embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")

    return data, docs_embeddings
