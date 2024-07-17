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
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import streamlit as st
import numpy as np
from tqdm import tqdm
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
import openai
import os
from pathlib import Path
import json


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
COMMON_NGRAMS = [
    "éléctricité",
    "RTE",
    "France",
    "électrique",
    "projet",
    "année",
    "transport électricité",
    "réseau électrique",
    "gestionnaire réseau",
    "réseau transport",
    "production électricité",
    "milliards euros",
    "euros",
    "2022",
    "2023",
    "2024",
    "électricité RTE",
    "Réseau transport",
    "RTE gestionnaire",
    "électricité France",
    "système électrique"
]

# Define the path to your JSON file
stopwords_fr_file = Path(os.getcwd()) / "wattelse" / "bertopic" / "weak_signals" / 'stopwords-fr.json'

# Read the JSON data from the file and directly assign it to the list
with open(stopwords_fr_file, 'r', encoding='utf-8') as file:
    FRENCH_STOPWORDS = json.load(file)

DEFAULT_STOP_WORDS = FRENCH_STOPWORDS + STOP_WORDS_RTE + COMMON_NGRAMS
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
    Implements batch processing for efficient memory usage and handles different input types.
    """

    def __init__(self, model_name, batch_size=5000):
        super().__init__()

        logger.info(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Handle the particular scenario of when max seq length in original model is abnormal (not a power of 2)
        if self.embedding_model.max_seq_length == 514:
            self.embedding_model.max_seq_length = 512

        self.name = model_name
        self.batch_size = batch_size
        logger.debug("Embedding model loaded")

    def embed(self, documents: Union[List[str], pd.Series], verbose=True) -> np.ndarray:
        # Convert to list if input is a pandas Series
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        num_documents = len(documents)
        num_batches = (num_documents + self.batch_size - 1) // self.batch_size
        embeddings = []

        for i in tqdm(range(num_batches), desc="Embedding batches", disable=not verbose):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_documents)
            batch_documents = documents[start_idx:end_idx]
            
            batch_embeddings = self.embedding_model.encode(
                batch_documents,
                show_progress_bar=False,
                output_value=None
            )
            embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Embedded {num_documents} documents in {num_batches} batches")
        return all_embeddings


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
    
    # Separate OpenAI from other representation models
    openai_model = None
    other_rep_models = []
    for model in representation_models:
        if model == "OpenAI":
            openai_model = OpenAI(
                client=openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
                model=kwargs.get("openai_model", "gpt-3.5-turbo"),
                nr_docs=kwargs.get("openai_nr_docs", 5),
                prompt=FRENCH_CHAT_PROMPT if kwargs.get("data_language", "Français") == "Français" else None,
                chat=True
            )
        else:
            other_rep_models.append(model)

    # Initialize other representation models
    representation_model_objects = []
    for model in other_rep_models:
        if model == "MaximalMarginalRelevance":
            representation_model_objects.append(MaximalMarginalRelevance(
                diversity=kwargs.get("mmr_diversity", 0.3),
                top_n_words=kwargs.get("mmr_top_n_words", 10)
            ))
        elif model == "KeyBERTInspired":
            representation_model_objects.append(KeyBERTInspired(
                top_n_words=kwargs.get("keybert_top_n_words", 10),
                nr_repr_docs=kwargs.get("keybert_nr_repr_docs", 5),
                nr_candidate_words=kwargs.get("keybert_nr_candidate_words", 20)
            ))

    # Use the first model if only one is specified, otherwise use the list
    representation_model = representation_model_objects[0] if len(representation_model_objects) == 1 else representation_model_objects

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

    logger.info("Reducing outliers via embeddings strategy...")
    new_topics = topic_model.reduce_outliers(documents=filtered_dataset[column], 
                                             topics=topics, 
                                             embeddings=embeddings,
                                             strategy="embeddings")
    
    # WARNING : We have to repass the vectorizer and representation models again otherwise BERTopic will use the default ones
    topic_model.update_topics(filtered_dataset[column], topics=new_topics, 
                              vectorizer_model=vectorizer_model, 
                              representation_model=representation_model)
    
    # If OpenAI model is present, apply it after reducing outliers
    if openai_model:
        logger.info("Applying OpenAI representation model...")
        
        # WARNING : update_topics updates the topic model's representation model
        # Solution : Backup the current representation model(s) to avoid using OpenAI later during the computation of topics over time
        backup_representation_model = topic_model.representation_model 
        topic_model.update_topics(filtered_dataset[column], topics=new_topics, representation_model=openai_model)
        topic_model.representation_model = backup_representation_model 
    
    return topic_model, new_topics, probs, embeddings, token_embeddings, token_strings





def group_tokens(tokenizer, token_ids, token_embeddings, language="french"):
    grouped_token_lists = []
    grouped_embedding_lists = []
    
    special_tokens = {"english": ["[CLS]", "[SEP]", "[PAD]"], "french": ["<s>", "</s>", "<pad>"]}
    subword_prefix = {"english": "##", "french": "▁"}
    
    for token_id, token_embedding in tqdm(zip(token_ids, token_embeddings), desc="Grouping split tokens into whole words"):
        tokens = tokenizer.convert_ids_to_tokens(token_id)
        
        grouped_tokens = []
        grouped_embeddings = []
        current_word = ""
        current_embedding = []
        
        for token, embedding in zip(tokens, token_embedding):
            if token in special_tokens[language]:
                continue
            
            if language == "french" and token.startswith(subword_prefix[language]):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token[1:]
                current_embedding = [embedding]
            elif language == "english" and not token.startswith(subword_prefix[language]):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token
                current_embedding = [embedding]
            else:
                current_word += token.lstrip(subword_prefix[language])
                current_embedding.append(embedding)
        
        if current_word:
            grouped_tokens.append(current_word)
            grouped_embeddings.append(np.mean(current_embedding, axis=0))
        
        grouped_token_lists.append(grouped_tokens)
        grouped_embedding_lists.append(np.array(grouped_embeddings))
    
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
