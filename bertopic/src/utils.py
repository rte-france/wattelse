import gzip
import hashlib
import pickle
import socket
from pathlib import Path
from typing import Any, List

import nltk
import pandas as pd
from loguru import logger

nltk.download("stopwords")

DATA_DIR = (
    "/data/weak_signals/data/bertopic/"
    if socket.gethostname() == "groesplu0"
    else "cache"
)
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
GROUPED_TIMESTAMP_COLUMN = "grouped_timestamp"
URL_COLUMN = "url"
TITLE_COLUMN = "title"
CITATION_COUNT_COL = "citation_count"
BASE_CACHE_PATH = (
    Path("/data/weak_signals/cache/bertopic/")
    if socket.gethostname() == "groesplu0"
    else Path("cache")
)


def file_to_pd(file_name: str, base_dir: str = None) -> pd.DataFrame:
    """Read data in various format and convert in to a DataFrame"""
    data_path = base_dir + file_name if base_dir else file_name
    if ".csv" in file_name:
        return pd.read_csv(data_path)
    elif ".jsonl" in file_name or ".jsonlines" in file_name:
        return pd.read_json(data_path, lines=True)
    elif ".jsonl.gz" in file_name or ".jsonlines.gz" in file_name:
        with gzip.open(file_name) as f_in:
            return pd.read_json(f_in, lines=True)


def clean_dataset(dataset: pd.DataFrame, length_criteria: int):
    """Clean dataset. So far, only removes short text."""
    cleaned_dataset = dataset.loc[
        dataset[TEXT_COLUMN].str.len() >= length_criteria
    ].reset_index(drop=True)
    return cleaned_dataset


def load_embeddings(cache_path: Path):
    """Loads embeddings as pickle"""
    with open(cache_path, "rb") as f_in:
        return pickle.load(f_in)


def save_embeddings(embeddings: List, cache_path: Path):
    """Save embeddings as pickle"""
    with open(cache_path, "wb") as f_out:
        pickle.dump(embeddings, f_out)


def get_hash(data: Any):
    """Returns a *stable* hash(persistent between different Python session) for any object. NB. The default hash() function does not guarantee this."""
    return hashlib.md5(repr(data).encode("utf-8")).hexdigest()

def split_df_by_paragraphs(dataset: pd.DataFrame):
    """Split texts into multiple paragraphs and returns a concatenation of all extracts as a new pandas DF"""
    dataset[TEXT_COLUMN] = dataset[TEXT_COLUMN].str.split("\n")
    dataset = dataset.explode(TEXT_COLUMN)
    dataset = dataset[dataset[TEXT_COLUMN]!=""].reset_index(drop=True)
    return dataset
