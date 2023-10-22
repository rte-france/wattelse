import pdb
from typing import List

import torch
import typer
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from wattelse.bertopic.utils import file_to_pd, TEXT_COLUMN, TIMESTAMP_COLUMN, BASE_CACHE_PATH, load_data, load_embeddings, save_embeddings, split_df_by_paragraphs

# Parameters:

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_N_WORDS = 10
DEFAULT_NR_TOPICS = 10
DEFAULT_NGRAM_RANGE = (1, 1)

DEFAULT_UMAP_MODEL = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
DEFAULT_HBSCAN_MODEL = HDBSCAN(
    min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True
)
DEFAULT_STOP_WORDS = stopwords.words("french")
DEFAULT_VECTORIZER_MODEL = CountVectorizer(
    stop_words=DEFAULT_TOP_N_WORDS, ngram_range=DEFAULT_NGRAM_RANGE,
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
    
        logger.info(f"Loading model: {model_name} on device: {device}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.max_seq_length = 512
        self.name = model_name
        logger.info("Model loaded")

    def embed(self, documents: List[str], verbose=True) -> List:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


def train_BERTopic(
    texts: List[str],
    indexes: List[str],
    data_name: str,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    umap_model=DEFAULT_UMAP_MODEL,
    hdbscan_model=DEFAULT_HBSCAN_MODEL,
    vectorizer_model=DEFAULT_VECTORIZER_MODEL,
    ctfidf_model=DEFAULT_CTFIDF_MODEL,
    representation_model=DEFAULT_REPRESENTATION_MODEL,
    top_n_words=DEFAULT_TOP_N_WORDS,
    nr_topics=DEFAULT_NR_TOPICS,
    use_cache = True,
    split_by_paragraphs = False,
):

    if split_by_paragraphs:
        cache_path = BASE_CACHE_PATH / f"{embedding_model.name}_{data_name}_split_by_paragraphs.pkl"
    else:
        cache_path = BASE_CACHE_PATH / f"{embedding_model.name}_{data_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache: {use_cache}")
    if not cache_path.exists():
        logger.info("Computing embeddings")
        if split_by_paragraphs:
            full_df = split_df_by_paragraphs(load_data(data_name)).sort_values(by=TIMESTAMP_COLUMN, ascending=False).reset_index()
        else:
            full_df = load_data(data_name).sort_values(by=TIMESTAMP_COLUMN, ascending=False).reset_index()
        embeddings = embedding_model.embed(full_df[TEXT_COLUMN])
        save_embeddings(embeddings, cache_path)
        logger.info(f"Embeddings stored to cache file: {cache_path}")
    else:
        embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")
    
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
    topics, probs = topic_model.fit_transform(texts, embeddings[indexes])

    return topics, probs, topic_model


if __name__ == "__main__":
    app = typer.Typer()

    @app.command("train")
    def train(
        model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="name of the model to be used"),
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
