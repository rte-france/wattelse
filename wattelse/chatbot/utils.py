from pathlib import Path
from typing import List
import pickle
import sys
import socket

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GenerationConfig

from wattelse.common.vars import GPU_SERVERS

TEMPERATURE=0.1
MAX_TOKENS=512

BASE_PROMPT = """Répond à la question en utilisant le contexte fourni.
Contexte :
\"\"\"
{context}
\"\"\"
Question : {query}
La réponse doit être {expected_answer_size} et se baser uniquement sur le contexte. Si le contexte ne contient pas d'éléments permettant de répondre à la question, répondre \"Le contexte ne fourni pas assez d'information pour répondre à la question.\""""

BASE_CACHE_PATH = (
    Path("/data/weak_signals/cache/chatbot")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "cache" / "chatbot"
)

def make_docs_embedding(docs: List[str], embedding_model: SentenceTransformer):
    return embedding_model.encode(docs, show_progress_bar=True)


def extract_n_most_relevant_extracts(n, query, docs, docs_embeddings, embedding_model, similarity_threshold:float = 0):
    query_embedding = embedding_model.encode(query)
    similarity = cosine_similarity([query_embedding], docs_embeddings)[0]

    # Find indices of documents with similarity above threshold
    above_threshold_indices = np.where(similarity > similarity_threshold)[0]

    # Sort above-threshold indices by similarity and select top n
    max_indices = above_threshold_indices[np.argsort(similarity[above_threshold_indices])][-n:][::-1]

    return docs[max_indices].tolist(), similarity[max_indices].tolist()

def generate_RAG_prompt(query: str, context_elements: List[str], expected_answer_size="short", custom_prompt=None) -> str:
    """
    Generates RAG prompt using query and context.
    """
    context = "\n".join(context_elements)
    expected_answer_size = "courte" if expected_answer_size=="short" else "détaillée"
    if custom_prompt:
        return custom_prompt.format(context=context, query=query, expected_answer_size=expected_answer_size)
    else:
        return BASE_PROMPT.format(context=context, query=query, expected_answer_size=expected_answer_size)


def generate_answer_locally(instruct_model, tokenizer, prompt) -> str:
    """Uses the local model to generate the answer"""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        instruct_model.device
    )
    input_length = input_ids.shape[1]
    generated_outputs = instruct_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=1.0,
            max_new_tokens=MAX_TOKENS,
        ),
        return_dict_in_generate=True,
    )
    generated_tokens = generated_outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.debug(f"Answer: {generated_text}")
    return generated_text


def load_embeddings(cache_path: Path):
    """Loads embeddings as pickle"""
    with open(cache_path, "rb") as f_in:
        return pickle.load(f_in)


def save_embeddings(embeddings: List, cache_path: Path):
    """Save embeddings as pickle"""
    with open(cache_path, "wb") as f_out:
        pickle.dump(embeddings, f_out)

def load_data(
    data_file: Path, embedding_model: SentenceTransformer, use_cache: bool = True
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

    cache_path = BASE_CACHE_PATH / f"{data_file.name}.pkl"
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