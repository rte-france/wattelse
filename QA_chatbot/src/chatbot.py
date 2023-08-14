from pathlib import Path

import pandas as pd
import sys
import torch
import typer
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import make_docs_embedding, extract_n_most_relevant_extracts, generate_answer

### Parameters ###
DEFAULT_DATA_FILE = "./data/BP-2019.csv"
EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"
INSTRUCT_MODEL_NAME = "bofenghuang/vigogne-2-7b-instruct"
N = 5 # number of top relevant extracts to include as context in the prompt


def initialize_models():
    """Load models"""
    logger.info("Initializing models...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_NAME, padding_side="right", use_fast=False)
    instruct_model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return embedding_model, tokenizer, instruct_model

def load_data(data_file: Path, embedding_model: SentenceTransformer):
    """Loads data and transform them into embeddings"""
    logger.info(f"Using data from: {data_file}")
    data = pd.read_csv(data_file)
    docs = data["text"]
    docs_embeddings = make_docs_embedding(docs, embedding_model)
    return docs, docs_embeddings

def chat(data_file: Path = DEFAULT_DATA_FILE):
    # initialize models
    embedding_model, tokenizer, instruct_model = initialize_models()

    # load data
    docs, docs_embeddings = load_data(data_file, embedding_model)

    logger.debug("Ready to chat!")
    # Chatbot
    while True:
        query = input("Question: ")
        if query in ["bye", "exit", "ciao", "quit"]:
            sys.exit(0)
        relevant_extracts = extract_n_most_relevant_extracts(
            N, query, docs, docs_embeddings, embedding_model
        )
        generate_answer(instruct_model, tokenizer, query, relevant_extracts)


if __name__ == "__main__":
    typer.run(chat)
