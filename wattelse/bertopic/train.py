#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pdb
from typing import List, Tuple, Union

import pandas as pd
import torch
import typer
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import numpy as np
import streamlit as st
import numpy as np
from tqdm import tqdm
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
import openai
import os


from wattelse.bertopic.utils import (
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    file_to_pd,
)
from wattelse.common import BASE_CACHE_PATH
from wattelse.common.cache_utils import load_embeddings, save_embeddings, get_hash

# Parameters:

DEFAULT_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_N_WORDS = 10
DEFAULT_NR_TOPICS = 10
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 2

DEFAULT_UMAP_MODEL = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
DEFAULT_HBSCAN_MODEL = HDBSCAN(
    min_cluster_size=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

STOP_WORDS_RTE = ["w", "kw", "mw", "gw", "tw", "wh", "kwh", "mwh", "gwh", "twh", "volt", "volts", "000"]
DEFAULT_STOP_WORDS = stopwords.words("french") + STOP_WORDS_RTE
DEFAULT_VECTORIZER_MODEL = CountVectorizer(
    stop_words=DEFAULT_STOP_WORDS,
    ngram_range=DEFAULT_NGRAM_RANGE,
    min_df=DEFAULT_MIN_DF,
)
DEFAULT_CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)

DEFAULT_REPRESENTATION_MODEL = MaximalMarginalRelevance(diversity=0.3)

# Add this constant for the French prompt
FRENCH_CHAT_PROMPT = """
J'ai un topic qui contient les documents suivants :
[DOCUMENTS]
Le topic est décrit par les mots-clés suivants : [KEYWORDS]
Sur la base des informations ci-dessus, extraire une courte étiquette de topic dans le format suivant :
Topic : <étiquette du sujet>
"""

class EmbeddingModel(BaseEmbedder):
    """
    Custom class for the embedding model. Currently supports SentenceBert models (model_name should refer to a SentenceBert model).
    """

    def __init__(self, model_name):
        super().__init__()

        logger.info(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Handle the particular scenario of when max seq length in original model is abnormal (not a power of 2)
        if self.embedding_model.max_seq_length == 514: self.embedding_model.max_seq_length = 512

        self.name = model_name
        logger.debug("Embedding model loaded")

    def embed(self, documents: List[str], verbose=True) -> List:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose, output_value=None)
        return embeddings



def convert_to_numpy(obj, type=None):
    if isinstance(obj, torch.Tensor):
        if type=='token_id': return obj.numpy().astype(np.int64)
        else : return obj.numpy().astype(np.float32)
        
    elif isinstance(obj, list):
        return [convert_to_numpy(item) for item in obj]
    
    else:
        raise TypeError("Object must be a list or torch.Tensor")



def train_BERTopic(
    full_dataset: pd.DataFrame,
    indices: pd.Series = None,
    column: str = TEXT_COLUMN,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    umap_model: UMAP = DEFAULT_UMAP_MODEL,
    hdbscan_model: HDBSCAN = DEFAULT_HBSCAN_MODEL,
    vectorizer_model: CountVectorizer = DEFAULT_VECTORIZER_MODEL,
    ctfidf_model: ClassTfidfTransformer = DEFAULT_CTFIDF_MODEL,
    representation_models: List[str] = ["MaximalMarginalRelevance"],
    top_n_words: int = DEFAULT_TOP_N_WORDS,
    nr_topics: (str | int) = DEFAULT_NR_TOPICS,
    use_cache: bool = True,
    cache_base_name: str = None,
    **kwargs
) -> Tuple[BERTopic, List[int], ndarray, ndarray, List[ndarray], List[List[str]]]:
    """
    Train a BERTopic model with customizable representation models.

    Parameters:
    ----------
    full_dataset: pd.Dataframe
        The full dataset to train
    indices:  pd.Series
        Indices of the full_dataset to be used for partial training (ex. selection of date range of the full dataset)
        If set to None, use all indices
    column: str
        Column name containing the text data
    embedding_model_name: str
        Name of the embedding model to use
    umap_model: UMAP or TSNE
        UMAP or TSNE model to be used in BERTopic
    hdbscan_model: HDBSCAN
        HDBSCAN model to be used in BERTopic
    vectorizer_model: CountVectorizer
        CountVectorizer model to be used in BERTopic
    ctfidf_model:  ClassTfidfTransformer
        ClassTfidfTransformer model to be used in BERTopic
    representation_models: List[str]
        List of representation models to use
    keybert_nr_repr_docs: int
        Number of representative documents for KeyBERTInspired
    keybert_nr_candidate_words: int
        Number of candidate words for KeyBERTInspired
    keybert_top_n_words: int
        Top N words for KeyBERTInspired
    mmr_diversity: float
        Diversity parameter for MaximalMarginalRelevance
    mmr_top_n_words: int
        Top N words for MaximalMarginalRelevance
    openai_model: str
        OpenAI model to use
    openai_nr_docs: int
        Number of documents for OpenAI
    data_language: str
        Language of the data (default is "Français")
    top_n_words: int
        Number of descriptive words per topic
    nr_topics: int
        Number of topics
    use_cache: bool
        Parameter to decide to store embeddings of the full dataset in cache
    cache_base_name: str
        Base name of the cache (for easier identification). If not name is provided, one will be created based on the full_dataset text

    Returns:
    -------
    A tuple containing:
        - a topic model
        - a list of topics
        - an array of probabilities
        - the document embeddings
        - the token embeddings
        - the token strings
    """

    if use_cache and cache_base_name is None:
        cache_base_name = get_hash(full_dataset[column])

    cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{cache_base_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Using cache: {use_cache}")
    embedding_model = EmbeddingModel(embedding_model_name)

    # Filter the dataset based on the provided indices
    if indices is not None:
        filtered_dataset = full_dataset[full_dataset.index.isin(indices)].reset_index(drop=True)
    else:
        filtered_dataset = full_dataset
        
    if cache_path.exists() and use_cache:
        # use previous cache
        embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")
        token_embeddings = None
        token_strings = None
    else:
        logger.info("Computing embeddings")
        output = embedding_model.embed(filtered_dataset[column])
        
        embeddings = [item['sentence_embedding'].detach().cpu() for item in output]
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.numpy()
        
        token_embeddings = [item['token_embeddings'].detach().cpu() for item in output]
        token_ids = [item['input_ids'].detach().cpu() for item in output]
        
        token_embeddings = convert_to_numpy(token_embeddings)
        token_ids = convert_to_numpy(token_ids, type='token_id')
        
        tokenizer = embedding_model.embedding_model._first_module().tokenizer

        token_strings, token_embeddings = group_tokens(tokenizer, token_ids, token_embeddings, language="french")

        if use_cache:
            save_embeddings(embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")

    if nr_topics == 0: nr_topics = None
    
    # Prepare representation models
    rep_models = []
    for model in representation_models:
        if model == "KeyBERTInspired":
            rep_models.append(KeyBERTInspired(
                nr_repr_docs=kwargs.get(f"{model}_nr_repr_docs", 5),
                nr_candidate_words=kwargs.get(f"{model}_nr_candidate_words", 40),
                top_n_words=kwargs.get(f"{model}_top_n_words", 20)
            ))
        elif model == "MaximalMarginalRelevance":
            rep_models.append(MaximalMarginalRelevance(
                diversity=kwargs.get(f"{model}_diversity", 0.2),
                top_n_words=kwargs.get(f"{model}_top_n_words", 10)
            ))
        elif model == "OpenAI":
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            prompt = FRENCH_CHAT_PROMPT if kwargs.get(f"{model}_language", "Français") == "Français" else None
            rep_models.append(OpenAI(
                client=client,
                model=kwargs.get(f"{model}_model", "gpt-3.5-turbo"),
                nr_docs=kwargs.get(f"{model}_nr_docs", 5),
                prompt=prompt
            ))

    representation_model = rep_models[0] if len(rep_models) == 1 else rep_models

    # Build BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model.embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=top_n_words,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info("Fitting BERTopic...")
    
    topics, probs = topic_model.fit_transform(
        filtered_dataset[column],
        embeddings
    )
    
    return topic_model, topics, probs, embeddings, token_embeddings, token_strings



def group_tokens(tokenizer, token_ids, token_embeddings, language="english"):
    if language not in ["english", "french"]:
        raise ValueError("Invalid language. Supported languages: 'english', 'french'")
    
    special_tokens = {
        "english": ["[CLS]", "[SEP]", "[PAD]"],
        "french": ["<s>", "</s>", "<pad>"]
    }
    
    grouped_token_lists = []
    grouped_embedding_lists = []
    
    for token_id, token_embedding in tqdm(zip(token_ids, token_embeddings), desc="Processing documents"):
        tokens, embeddings = remove_special_tokens(tokenizer, token_id, token_embedding, special_tokens[language])
        grouped_tokens, grouped_embeddings = group_subword_tokens(tokens, embeddings, language)
        
        grouped_token_lists.append(grouped_tokens)
        grouped_embedding_lists.append(np.stack(grouped_embeddings))  # Stack to form numpy arrays
    
    return grouped_token_lists, grouped_embedding_lists

def remove_special_tokens(tokenizer, token_id, token_embedding, special_tokens):
    tokens = tokenizer.convert_ids_to_tokens(token_id)
    
    filtered_tokens = []
    filtered_embeddings = []
    for token, embedding in zip(tokens, token_embedding):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_embeddings.append(embedding)
    
    return filtered_tokens, filtered_embeddings

def group_subword_tokens(tokens, embeddings, language):
    grouped_tokens = []
    grouped_embeddings = []
    current_token = []
    current_embedding_sum = np.zeros_like(embeddings[0])
    num_subtokens = 0
    
    for token, embedding in zip(tokens, embeddings):
        if (language == "english" and token.startswith("##")) or (language == "french" and token.startswith("▁")):
            if language == "english":
                current_token.append(token[2:])
            else:  # French
                if current_token:
                    grouped_tokens.append("".join(current_token))
                    grouped_embeddings.append(current_embedding_sum / num_subtokens)
                current_token = [token[1:]]  # Remove the '▁' character
                current_embedding_sum = embedding
                num_subtokens = 1
                continue
        else:
            if current_token:
                grouped_tokens.append("".join(current_token))
                grouped_embeddings.append(current_embedding_sum / num_subtokens)
            current_token = []
            current_embedding_sum = np.zeros_like(embedding)
            num_subtokens = 0
        
        current_token.append(token)
        current_embedding_sum += embedding
        num_subtokens += 1
    
    if current_token:
        grouped_tokens.append("".join(current_token))
        grouped_embeddings.append(current_embedding_sum / num_subtokens)
    
    return grouped_tokens, grouped_embeddings





if __name__ == "__main__":
    app = typer.Typer()

    @app.command("train")
    def train(
        model: str = typer.Option(
            DEFAULT_EMBEDDING_MODEL_NAME, help="name of the model to be used"
        ),
        data_path: str = typer.Option(
            None, help="path to the data from which topics have to be analyzed"
        ),
        topics: int = typer.Option(10, help="number of topics"),
        save_model_path: str = typer.Option(None, help="where to save the model"),
    ):

        embedding_model = EmbeddingModel(model)

        # Load data
        logger.info(f"Loading data from: {data_path}...")
        data = file_to_pd(data_path)
        logger.info("Data loaded")

        # train BERTopic
        topics, probs, topic_model = train_BERTopic(
            embedding_model=embedding_model, texts=data[TEXT_COLUMN], nr_topics=topics
        )

        # save model
        if save_model_path:
            topic_model.save(save_model_path, serialization="pytorch")

        pdb.set_trace()


    app()
