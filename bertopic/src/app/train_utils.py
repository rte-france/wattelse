import ast

from typing import List
import pandas as pd
import streamlit as st
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from train import EmbeddingModel, train_BERTopic
from utils import TEXT_COLUMN


@st.cache_data
def train_BERTopic_wrapper(dataset: pd.DataFrame, form_parameters, data_name: str, split_by_paragraphs = False):

    # Transform form_parameters from str to dict (dict is not yet hashable using Streamlit)
    form_parameters = ast.literal_eval(form_parameters)

    # Step 1 - Embedding model
    embedding_model = EmbeddingModel(form_parameters["embedding_model_name"])

    # Step 2 - Dimensionality reduction algorithm
    umap_model = UMAP(
        n_neighbors=form_parameters["umap_n_neighbors"],
        n_components=form_parameters["umap_n_components"],
        min_dist=form_parameters["umap_min_dist"],
        metric=form_parameters["umap_metric"],
    )

    # Step 3 - Clustering algorithm
    hdbscan_model = HDBSCAN(
        min_cluster_size=form_parameters["hdbscan_min_cluster_size"],
        min_samples=form_parameters["hdbscan_min_samples"],
        metric=form_parameters["hdbscan_metric"],
        cluster_selection_method=form_parameters["hdbscan_cluster_selection_method"],
        prediction_data=True,
    )

    # Step 4 - Count vectorizer
    stop_words = (
        stopwords.words(form_parameters["countvectorizer_stop_words"])
        if form_parameters["countvectorizer_stop_words"]
        else None
    )

    vectorizer_model = CountVectorizer(
        stop_words=stop_words, ngram_range=form_parameters["countvectorizer_ngram_range"],
    )

    # Step 5 - c-TF-IDF model
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=form_parameters["ctfidf_reduce_frequent_words"]
    )

    return train_BERTopic(
        dataset[TEXT_COLUMN],
        dataset["index"],
        data_name,
        embedding_model,
        umap_model,
        hdbscan_model,
        vectorizer_model,
        ctfidf_model,
        top_n_words=form_parameters["bertopic_top_n_words"],
        nr_topics=form_parameters["bertopic_nr_topics"] if form_parameters["bertopic_nr_topics"] > 0 else None,
        use_cache=form_parameters["use_cached_embeddings"],
        split_by_paragraphs=split_by_paragraphs,
    )
