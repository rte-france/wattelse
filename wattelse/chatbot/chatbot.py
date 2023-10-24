import socket
from pathlib import Path
from typing import List

import pandas as pd
import pickle
import sys
import torch
import typer
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from wattelse.chatbot.utils import (
    make_docs_embedding,
    extract_n_most_relevant_extracts,
    generate_answer_locally,
    generate_answer_remotely,
    generate_RAG_prompt,
)
from wattelse.common.vars import GPU_SERVERS

### Parameters ###
DEFAULT_DATA_FILE = "./data/BP-2019.csv"
EMBEDDING_MODEL_NAME = "antoinelouis/biencoder-camembert-base-mmarcoFR"
INSTRUCT_MODEL_NAME = "bofenghuang/vigogne-2-7b-instruct"
N = 5  # number of top relevant extracts to include as context in the prompt

SPECIAL_CHARACTER = ">"

BASE_CACHE_PATH = (
    Path("/data/weak_signals/cache/chatbot")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "cache" / "chatbot"
)


def load_embeddings(cache_path: Path):
    """Loads embeddings as pickle"""
    with open(cache_path, "rb") as f_in:
        return pickle.load(f_in)


def save_embeddings(embeddings: List, cache_path: Path):
    """Save embeddings as pickle"""
    with open(cache_path, "wb") as f_out:
        pickle.dump(embeddings, f_out)


def initialize_models(use_remote_llm: bool = True):
    """Load models"""
    logger.info("Initializing models...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_model.max_seq_length = (
        512  # Default is 514, this creates error with big texts
    )
    if use_remote_llm:
        tokenizer = None
        instruct_model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            INSTRUCT_MODEL_NAME, padding_side="right", use_fast=False
        )
        instruct_model = AutoModelForCausalLM.from_pretrained(
            INSTRUCT_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
    return embedding_model, tokenizer, instruct_model


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


def chat(data_file: Path = DEFAULT_DATA_FILE, use_remote_llm: bool = False):
    # initialize models
    embedding_model, tokenizer, instruct_model = initialize_models(use_remote_llm)

    # load data
    docs, docs_embeddings = load_data(data_file, embedding_model, use_cache=True)

    logger.debug("Ready to chat!")
    # Chatbot
    while True:
        query = input("Question: ")
        if query in ["bye", "exit", "ciao", "quit"]:
            sys.exit(0)
        if query.startswith(SPECIAL_CHARACTER):
            # we use the previous answer as an additional context to continue the conversation thread
            if (
                "answer" in locals()
            ):  # the variable exists, set from previous interaction
                query = answer + " " + query[1:]
        relevant_extracts, _ = extract_n_most_relevant_extracts(
            N, query, docs, docs_embeddings, embedding_model
        )

        # Generates prompt
        prompt = generate_RAG_prompt(
            query, relevant_extracts
        )

        if use_remote_llm:
            answer = generate_answer_remotely(prompt)
        else:
            answer = generate_answer_locally(instruct_model, tokenizer, prompt)
        print(answer)


if __name__ == "__main__":
    typer.run(chat)
