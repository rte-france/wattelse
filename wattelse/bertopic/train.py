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
import numpy as np
import streamlit as st
import numpy as np
from tqdm import tqdm


from wattelse.bertopic.utils import (
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    BASE_CACHE_PATH,
    file_to_pd,
)
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
    representation_model: MaximalMarginalRelevance = DEFAULT_REPRESENTATION_MODEL,
    top_n_words: int = DEFAULT_TOP_N_WORDS,
    nr_topics: (str | int) = DEFAULT_NR_TOPICS,
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

    # Filter the dataset based on the provided indices
    if indices is not None:
        filtered_dataset = full_dataset[full_dataset.index.isin(indices)].reset_index(drop=True)
    else:
        filtered_dataset = full_dataset
    
    logger.debug(f"FILTERED DATA : {filtered_dataset.iloc[:5]['text'].tolist()}")
    
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

        # logger.debug("Document and token_ids before grouping:")
        # for i in range(len(filtered_dataset[column])):
        #     logger.debug(f"Document: {filtered_dataset[column][i]} LENGTH : {len(filtered_dataset[column][i])}")


        # logger.debug(f"BEFORE grouping: {len(token_embeddings)}")
        token_strings, token_embeddings = group_tokens(tokenizer, token_ids, token_embeddings, language="french")
        # logger.debug(f"AFTER grouping: {len(token_embeddings)}")

        if use_cache:
            save_embeddings(embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")


    if nr_topics == 0: nr_topics = "auto"
    
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
    
    if indices is None:
            indices = full_dataset.index
    topics, probs = topic_model.fit_transform(
        filtered_dataset[column],
        embeddings
    )
    
    return topic_model, topics, probs, embeddings, token_embeddings, token_strings



def group_tokens(tokenizer, token_ids, token_embeddings, language="english"):
    grouped_token_lists = []
    grouped_embedding_lists = []
    for token_id, token_embedding in tqdm(zip(token_ids, token_embeddings), desc="Processing documents"):
        # print(tokenizer.convert_ids_to_tokens(token_id))
        if language == "english":
            tokens, embeddings = remove_special_tokens_english(tokenizer, token_id, token_embedding)
            grouped_tokens, grouped_embeddings = group_subword_tokens_english(tokens, embeddings)

        elif language == "french":
            tokens, embeddings = remove_special_tokens_french(tokenizer, token_id, token_embedding)
            grouped_tokens, grouped_embeddings = group_subword_tokens_french(tokens, embeddings)
        else:
            raise ValueError("Invalid language. Supported languages: 'english', 'french'")
        
        grouped_token_lists.append(grouped_tokens)
        grouped_embedding_lists.append(np.stack(grouped_embeddings))  # Stack to form numpy arrays
    return grouped_token_lists, grouped_embedding_lists

def remove_special_tokens_english(tokenizer, token_id, token_embedding):
    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    tokens = tokenizer.convert_ids_to_tokens(token_id)
    
    filtered_tokens = []
    filtered_embeddings = []
    for token, embedding in zip(tokens, token_embedding):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_embeddings.append(embedding)
    
    return filtered_tokens, filtered_embeddings

def remove_special_tokens_french(tokenizer, token_id, token_embedding):
    special_tokens = ["<s>", "</s>", "<pad>"]
    tokens = tokenizer.convert_ids_to_tokens(token_id)
    
    filtered_tokens = []
    filtered_embeddings = []
    for token, embedding in zip(tokens, token_embedding):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_embeddings.append(embedding)
    
    return filtered_tokens, filtered_embeddings

def group_subword_tokens_english(tokens, embeddings):
    grouped_tokens = []
    grouped_embeddings = []
    current_token = []
    current_embedding_sum = np.zeros_like(embeddings[0])
    num_subtokens = 0
    for token, embedding in zip(tokens, embeddings):
        if token.startswith("##"):
            current_token.append(token[2:])
            current_embedding_sum += embedding  # Use numpy addition
            num_subtokens += 1
        else:
            if current_token:
                grouped_tokens.append("".join(current_token))
                grouped_embeddings.append(current_embedding_sum / num_subtokens)  # Use numpy division for average
            current_token = []
            current_embedding_sum = np.zeros_like(embedding)  # Reset with numpy zeros
            num_subtokens = 0
            current_token.append(token)
            current_embedding_sum += embedding
            num_subtokens += 1
    if current_token:
        grouped_tokens.append("".join(current_token))
        grouped_embeddings.append(current_embedding_sum / num_subtokens)  # Final averaging
    return grouped_tokens, grouped_embeddings

def group_subword_tokens_french(tokens, embeddings):
    grouped_tokens = []
    grouped_embeddings = []
    current_token = []
    current_embedding_sum = np.zeros_like(embeddings[0])
    num_subtokens = 0
    for token, embedding in zip(tokens, embeddings):
        if token.startswith("▁"):
            if current_token:
                grouped_tokens.append("".join(current_token))
                grouped_embeddings.append(current_embedding_sum / num_subtokens)  # Use numpy division for average
            current_token = [token[1:]]  # Remove the '▁' character
            current_embedding_sum = embedding
            num_subtokens = 1
        else:
            current_token.append(token)
            current_embedding_sum += embedding  # Use numpy addition
            num_subtokens += 1
    if current_token:
        grouped_tokens.append("".join(current_token))
        grouped_embeddings.append(current_embedding_sum / num_subtokens)  # Final averaging
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
