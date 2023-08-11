import gzip
import hashlib
import pickle

import pandas as pd
import nltk
from loguru import logger
from pathlib import Path
from typing import Any, List

nltk.download("stopwords")

DATA_DIR = "./data/"
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
BASE_CACHE_PATH = Path("cache")

def file_to_pd(file_name: str, base_dir: str = None) -> pd.DataFrame:
    data_path = base_dir + file_name if base_dir else file_name
    if ".csv" in file_name:
        return pd.read_csv(data_path)
    elif ".jsonl" in file_name or ".jsonlines" in file_name:
        return pd.read_json(data_path, lines=True)
    elif ".jsonl.gz" in file_name or ".jsonlines.gz" in file_name:
        with gzip.open(file_name) as f_in:
            return pd.read_json(f_in, lines=True)


def clean_dataset(dataset: pd.DataFrame, length_criteria: int):
    cleaned_dataset = dataset.loc[dataset[TEXT_COLUMN].str.len() >= length_criteria]
    logger.debug(f"Cleaned dataset reduced to: {len(cleaned_dataset)} items")
    return cleaned_dataset


def load_embeddings(cache_path: Path):
    with open(cache_path, "rb") as f_in:
        return pickle.load(f_in)

def save_embeddings(embeddings: List, cache_path: Path):
    with open(cache_path, "wb") as f_out:
        pickle.dump(embeddings, f_out)

def get_hash(data: Any):
    """Returns a *stable* hash(persistent between different Python session) for any object. NB. The default hash() function does not guarantee this."""
    return hashlib.md5(repr(data).encode("utf-8")).hexdigest()
