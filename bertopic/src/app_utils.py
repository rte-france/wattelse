import streamlit as st

from utils import TEXT_COLUMN, TIMESTAMP_COLUMN, DATA_DIR, file_to_pd


@st.cache_data
def load_data(data_name: str):
    return file_to_pd(data_name, base_dir=DATA_DIR)


def data_cleaning_options():
    return {
        "min_text_length": st.number_input("min_text_length (#chars)", min_value=0, value=300),
    }

def embedding_model_options():
    return {
        "embedding_model_name": st.selectbox("Name", ["paraphrase-multilingual-MiniLM-L12-v2"]),
        "use_cached_embeddings": st.checkbox("Put embeddings in cache", value=True)
    }


def bertopic_options():
    return {
        "bertopic_nr_topics": st.number_input("nr_topics", min_value=0, value=0),
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
        "countvectorizer_ngram_range": st.selectbox(
            "ngram_range", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        ),
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
def plot_topics_over_time(form_parameters, _topic_model, df, nr_bins=50, width=1000):
    topics_over_time = _topic_model.topics_over_time(
        df[TEXT_COLUMN], df[TIMESTAMP_COLUMN], nr_bins=nr_bins, global_tuning=False
    )
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
