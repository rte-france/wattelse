from pathlib import Path
from typing import List
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from wattelse.common.cache_utils import save_embeddings, load_embeddings
from wattelse.common.vars import BASE_CACHE_PATH
from wattelse.llm.prompts import FR_USER_BASE_RAG

CACHE_DIR = BASE_CACHE_PATH / "chatbot"
# Make dirs if not exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def make_docs_embedding(docs: List[str], embedding_model: SentenceTransformer):
    return embedding_model.encode(docs, show_progress_bar=True)


def extract_n_most_relevant_extracts(
    n, query, docs, docs_embeddings, embedding_model, similarity_threshold: float = 0
):
    query_embedding = embedding_model.encode(query)
    similarity = cosine_similarity([query_embedding], docs_embeddings)[0]

    # Find indices of documents with similarity above threshold
    above_threshold_indices = np.where(similarity > similarity_threshold)[0]

    # Sort above-threshold indices by similarity and select top n
    max_indices = above_threshold_indices[
        np.argsort(similarity[above_threshold_indices])
    ][-n:][::-1]

    return docs[max_indices].tolist(), similarity[max_indices].tolist()


def generate_RAG_prompt(
    query: str,
    context_elements: List[str],
    expected_answer_size="short",
    custom_prompt=None,
) -> str:
    """
    Generates RAG prompt using query and context.
    """
    context = "\n".join(context_elements)
    expected_answer_size = "courte" if expected_answer_size == "short" else "détaillée"
    if custom_prompt:
        return custom_prompt.format(
            context=context, query=query, expected_answer_size=expected_answer_size
        )
    else:
        return FR_USER_BASE_RAG.format(
            context=context, query=query, expected_answer_size=expected_answer_size
        )


def load_data(
    data_file: Path,
    embedding_model: SentenceTransformer,
    embedding_model_name: str = None,
    use_cache: bool = True,
):
    """Loads data and transform them into embeddings; data file shall contain a column 'processed_text' (preferred)
    or 'text'"""
    logger.info(f"Using data from: {data_file}")
    data = pd.read_csv(data_file, keep_default_na=False)
    if "processed_text" in data.columns:
        docs = data["processed_text"]
    elif "text" in data.columns:
        docs = data["text"]
    else:
        logger.error(
            f"Data {data_file} not formatted correctly! (expecting 'processed_text' or 'text' column"
        )
        sys.exit(-1)

    cache_path = CACHE_DIR / f"{embedding_model_name}_{data_file.name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache: {use_cache}")
    if not use_cache or not cache_path.exists():
        logger.info("Computing sentence embeddings...")
        docs_embeddings = make_docs_embedding(docs, embedding_model)
        if use_cache:
            save_embeddings(docs_embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")
    else:
        docs_embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")

    return docs, docs_embeddings
