import pdb

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.backend import BaseEmbedder

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import ast
import streamlit as st


@st.cache_data
def load_data(data_name):
	return pd.read_csv("./data/cleaned/"+data_name)


class EmbeddingModel(BaseEmbedder):
	"""
	Custom class for the embedding model. Currently supports SentenceBert models (model_name should refer to a SentenceBert model).
	"""
	def __init__(self, model_name):
		super().__init__()
		self.embedding_model = SentenceTransformer(model_name)

	def embed(self, documents, verbose=True):
		embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
		return embeddings 


@st.cache_data
def BERTopic_train(docs, form_parameters):

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
	stop_words = stopwords.words(form_parameters["countvectorizer_stop_words"]) if form_parameters["countvectorizer_stop_words"] else None
	
	vectorizer_model = CountVectorizer(
		stop_words=stop_words,
		ngram_range=form_parameters["countvectorizer_ngram_range"],
	)
	

	# Step 5 - c-TF-IDF model
	ctfidf_model = ClassTfidfTransformer(
		reduce_frequent_words=form_parameters["ctfidf_reduce_frequent_words"]
	)

	# Build BERTopic model
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=umap_model,
		hdbscan_model=hdbscan_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		top_n_words=form_parameters["bertopic_top_n_words"],
		nr_topics=form_parameters["bertopic_nr_topics"],
		calculate_probabilities=True,
		verbose=True,
	)
	
	topics, probs = topic_model.fit_transform(docs)
	
	return topics, probs, topic_model



def embedding_model_options():
	return {
		"embedding_model_name": st.selectbox("Name", ["paraphrase-multilingual-MiniLM-L12-v2"]),
	}

def bertopic_options():
	return {
		"bertopic_nr_topics": st.number_input("nr_topics", min_value=1, value=50),
	    "bertopic_top_n_words": st.number_input("top_n_words", min_value=1, value=10),
	}

def umap_options():
	return {
		"umap_n_neighbors": st.number_input("n_neighbors", min_value=1, value=15),
	    "umap_n_components": st.number_input("n_components", min_value=1, value=5),
		"umap_min_dist": st.number_input("min_dist", min_value=0.0, value=0.0, max_value=1.0),
		"umap_metric": st.selectbox("metric", ["cosine"]),
	}

def hdbscan_options():
	return {
		"hdbscan_min_cluster_size": st.number_input("min_cluster_size", min_value=1, value=10),
		"hdbscan_min_samples": st.number_input("min_samples", min_value=1, value=10),
		"hdbscan_metric": st.selectbox("metric", ["euclidean"]),
		"hdbscan_cluster_selection_method": st.selectbox("cluster_selection_method", ["eom"]),
	}

def countvectorizer_options():
	return {
		"countvectorizer_stop_words": st.selectbox("stop_words", ["french", None]),
		"countvectorizer_ngram_range": st.selectbox("ngram_range", [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]),
	}

def ctfidf_options():
	return {
		"ctfidf_reduce_frequent_words": st.selectbox("reduce_frequent_words", [True, False]),
	}


@st.cache_data
def plot_barchart(form_parameters, _topic_model, width=200):
	return _topic_model.visualize_barchart(width=width)

@st.cache_data
def plot_topics_hierarchy(form_parameters, _topic_model, width=800):
	return _topic_model.visualize_hierarchy(width=width)

@st.cache_data
def plot_topics_over_time(form_parameters, _topic_model, df, nr_bins = 50, width=1000):
	topics_over_time = _topic_model.topics_over_time(df["docs"], df["timestamps"], nr_bins=nr_bins, global_tuning=False)
	return _topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, width=width)


def print_search_results(search_term, topic_model):
	if search_term != "":
		topics, similarity = topic_model.find_topics(search_term=search_term, top_n=5)
		main_topic_infos = topic_model.get_topic_info(topics[0])
		st.write("### Most related topic:")
		st.write(main_topic_infos)
		st.write("### A representative article of this topic:")
		st.write(topic_model.get_representative_docs(topic=topics[0])[0])

	
def print_new_document_probs(new_document, topic_model):
	if new_document != "":
		topics, probs = topic_model.transform(new_document)
		st.write(topic_model.visualize_distribution(probs.squeeze()))