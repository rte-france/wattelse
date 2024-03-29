#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pdb
from typing import List, Tuple

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
DEFAULT_NGRAM_RANGE = (1, 1)
DEFAULT_MIN_DF = 2

DEFAULT_UMAP_MODEL = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
DEFAULT_HBSCAN_MODEL = HDBSCAN(
    min_cluster_size=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
DEFAULT_STOP_WORDS = stopwords.words("french")
DEFAULT_VECTORIZER_MODEL = CountVectorizer(
    stop_words=DEFAULT_STOP_WORDS,
    ngram_range=DEFAULT_NGRAM_RANGE,
    min_df=DEFAULT_MIN_DF,
)
DEFAULT_CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)

DEFAULT_REPRESENTATION_MODEL = MaximalMarginalRelevance(diversity=0.3)


class EmbeddingModel(BaseEmbedder):
    """
    Custom class for the embedding model. Currently supports SentenceBert models (model_name should refer to a SentenceBert model).
    """

    def __init__(self, model_name):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.max_seq_length = 512
        self.name = model_name
        logger.debug("Embedding model loaded")

    def embed(self, documents: List[str], verbose=True) -> List:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


def train_BERTopic(
    full_dataset: pd.DataFrame,
    indices: pd.Series = None,
    column: str = TEXT_COLUMN,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    umap_model: UMAP = DEFAULT_UMAP_MODEL,
    hdbscan_model: HDBSCAN = DEFAULT_HBSCAN_MODEL,
    vectorizer_model: CountVectorizer = DEFAULT_VECTORIZER_MODEL,
    ctfidf_model: ClassTfidfTransformer = DEFAULT_CTFIDF_MODEL,
    representation_model: MaximalMarginalRelevance = DEFAULT_REPRESENTATION_MODEL,
    top_n_words: int = DEFAULT_TOP_N_WORDS,
    nr_topics: int = DEFAULT_NR_TOPICS,
    use_cache: bool = True,
    cache_base_name: str = None,
) -> Tuple[BERTopic, List[int], ndarray]:
    """
    Train a BERTopic model

    Parameters
    ----------
    full_dataset: pd.Dataframe
        The full dataset to train
    indices:  pd.Series
        Indices of the full_dataset to be used for partial training (ex. selection of date range of the full dataset)
        If set to None, use all indices
    embedding_model: EmbeddingModel
        Embedding model to be used in BERTopic
    umap_model: UMAP
        UMAP model to be used in BERTopic
    hdbscan_model: HDBSCAN
        HDBSCAN model to be used in BERTopic
    vectorizer_model: CountVectorizer
        CountVectorizer model to be used in BERTopic
    ctfidf_model:  ClassTfidfTransformer
        ClassTfidfTransformer model to be used in BERTopic
    representation_model: MaximalMarginalRelevance
        Representation model to be used in BERTopic
    top_n_words: int
        Number of descriptive words per topic
    nr_topics: int
        Number of topics
    use_cache: bool
        Parameter to decide to store embeddings of the full dataset in cache
    cache_base_name: str
        Base name of the cache (for easier identification). If not name is provided, one will be created based on the full_dataset text

    Returns
    -------
    A tuple made of:
        - a topic model
        - a list of topics
        - an array of probabilities
    """

    if use_cache and cache_base_name is None:
        cache_base_name = get_hash(full_dataset[column])

    cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{cache_base_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Using cache: {use_cache}")
    embedding_model = EmbeddingModel(embedding_model_name)
    if cache_path.exists() and use_cache:
        # use previous cache
        embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")
    else:
        logger.info("Computing embeddings")
        embeddings = embedding_model.embed(full_dataset[column])
        if use_cache:
            save_embeddings(embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")

    # Build BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
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
    if indices is None:
        indices = full_dataset.index
    topics, probs = topic_model.fit_transform(
        full_dataset[column][indices], embeddings[indices]
    )

    return topic_model, topics, probs


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
