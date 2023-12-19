from pathlib import Path
from typing import List, Tuple
import sys
import string

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import Stemmer
from rank_bm25 import BM25Plus

from wattelse.common import TEXT_COLUMN
from wattelse.common.cache_utils import save_embeddings, load_embeddings
from wattelse.common.vars import BASE_CACHE_PATH
from wattelse.llm.prompts import FR_USER_BASE_RAG

CACHE_DIR = BASE_CACHE_PATH / "chatbot"
# Make dirs if not exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STEMMER = Stemmer.Stemmer("french")

def bm25_preprocessing(text : str):
    """Text preprocessing function before sending it to BM25"""
    text = text.lower() # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    tokenized_text = text.split(" ") # 1 token = 1 word
    tokenized_text = list(filter(None, tokenized_text)) # remove empty strings
    tokenized_text = STEMMER.stemWords(tokenized_text) # lemmatize tokens
    return tokenized_text

def make_docs_BM25_indexing(docs: List[str]):
    """Index a list of docs using BM25+ algorithm"""
    tokenized_docs = [bm25_preprocessing(doc) for doc in docs]
    bm25 = BM25Plus(tokenized_docs)
    return bm25

def compute_bm25_score(query: str, bm25_model: BM25Plus):
    """Computes BM25 score according to a query"""
    query = bm25_preprocessing(query)
    return bm25_model.get_scores(query)

def make_docs_embedding(docs: pd.DataFrame, embedding_model: SentenceTransformer):
    """Embedds a list of docs using a SentenceTransformer model"""
    return embedding_model.encode(docs, show_progress_bar=True)

def compute_dense_embeddings_score(query: str, docs_embeddings: np.ndarray, embedding_model: SentenceTransformer):
    """Computes similarity score between a query and docs dense embeddings"""
    query_embedding = embedding_model.encode(query)
    return cosine_similarity([query_embedding], docs_embeddings)[0]

def rerank_extracts(query: str, docs: pd.DataFrame, top_n: int, reranker_model: CrossEncoder, similarity_threshold: float = 0):
    """Rerank extract using a CrossEncoder model. Used after an hybrid search over documents (mode hybrid+reranker).

    Args:
        query (str): query
        docs (pd.DataFrame): corpus of documents
        top_n (int): top n extracts to return
        reranker_model (CrossEncoder): reranker model
        similarity_threshold (float, optional): extracts with similarity score lower
                                                than similarity_threshold will be discarded

    Returns:
        Tuple[List[str], List[float]]: list of relevant extracts and their associated similarity score
    """
    pairs = [(query, doc) for doc in docs]
    similarity_score = reranker_model.predict(pairs)
    above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
    top_n_indices = above_threshold_indices[np.argsort(similarity_score[above_threshold_indices])][::-1][:top_n]
    relevant_extracts = docs[top_n_indices].tolist()
    relevant_extracts_scores = similarity_score[top_n_indices].tolist()
    return relevant_extracts, relevant_extracts_scores

def extract_n_most_relevant_extracts(top_n: int,
                                     query: str,
                                     data: pd.DataFrame,
                                     docs_embeddings: np.ndarray,
                                     embedding_model: SentenceTransformer,
                                     bm25_model: BM25Plus,
                                     retrieval_mode: str = "dense",
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
        embedding_model (SentenceTransformer): embedding model
        bm25_model (BM25Plus): BM25 model
        retrieval_mode (str, optional): algorithm to retrieve top_n extracts :
                                        - 'bm25' -> uses BM25 model
                                        - 'dense' -> uses embedding_model
                                        - 'hybrid' -> uses both bm25 and embedding_model
                                        - 'hybrid+reranker' -> uses uses both bm25 and embedding_model and rerank using crossencoder
        reranker_model (CrossEncoder, optional): reranker model
        reranker_ratio (int, optional): reranker_ratio*top_n extracts are send to the reranker model 
        similarity_threshold (float, optional): extracts with similarity score lower
                                                than similarity_threshold will be discarded

    Returns:
        Tuple[List[str], List[float]]: list of relevant extracts and their associated similarity score
    """
    # BM25 model
    if retrieval_mode == "bm25":
        similarity_score = compute_bm25_score(query, bm25_model)
        above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
        top_n_indices = above_threshold_indices[np.argsort(similarity_score[above_threshold_indices])][::-1][:top_n]
        relevant_extracts = docs[top_n_indices].tolist()
        relevant_extracts_similarity = similarity_score[top_n_indices].tolist()
    
    # Dense model
    elif retrieval_mode == "dense":
        similarity_score = compute_dense_embeddings_score(query, docs_embeddings, embedding_model)
        above_threshold_indices = np.where(similarity_score > similarity_threshold)[0]
        top_n_indices = above_threshold_indices[np.argsort(similarity_score[above_threshold_indices])][::-1][:top_n]
        relevant_extracts = docs[top_n_indices].tolist()
        relevant_extracts_similarity = similarity_score[top_n_indices].tolist()
    
    # Hybrid
    elif retrieval_mode == "hybrid":
        dense_similarity_score = compute_dense_embeddings_score(query, docs_embeddings, embedding_model)
        bm25_similarity_score = compute_bm25_score(query, bm25_model)
        sorted_dense_indices = dense_similarity_score.argsort()[::-1]
        sorted_bm25_indices = bm25_similarity_score.argsort()[::-1]
        logger.debug(f"Nombre d'extraits communs : {len(np.intersect1d(sorted_dense_indices[:top_n], sorted_bm25_indices[:top_n]))}")
        # Interleave dense and bm25 top indices to get half from each algorithm
        interleaved_indices = np.vstack((sorted_dense_indices, sorted_bm25_indices)).reshape((-1,),order='F')
        # Remove duplicates
        interleaved_indices = pd.unique(interleaved_indices)
        top_n_indices = interleaved_indices[:top_n]
        relevant_extracts = docs[top_n_indices].tolist()
        relevant_extracts_similarity = np.zeros(len(relevant_extracts)).tolist() # TODO : get scores for each extract

    # Hybrid + Reranker
    elif retrieval_mode == "hybrid+reranker":
        dense_similarity_score = compute_dense_embeddings_score(query, docs_embeddings, embedding_model)
        bm25_similarity_score = compute_bm25_score(query, bm25_model)
        top_n_dense_indices = dense_similarity_score.argsort()[::-1][:top_n*reranker_ratio]
        top_n_bm25_indices = bm25_similarity_score.argsort()[::-1][:top_n*reranker_ratio]
        relevant_extracts = pd.concat([docs[top_n_dense_indices], docs[top_n_bm25_indices]])
        relevant_extracts = relevant_extracts.drop_duplicates(keep = "first").reset_index(drop=True) # remove duplicates
        relevant_extracts, relevant_extracts_similarity = rerank_extracts(query,
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
            context=context, query=query, expected_answer_size=expected_answer_size, history=history,
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
        docs_embeddings = make_docs_embedding(docs, embedding_model)
        if use_cache:
            save_embeddings(docs_embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")
    else:
        docs_embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")

    return data, docs_embeddings
