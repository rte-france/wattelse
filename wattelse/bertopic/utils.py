#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import gzip
import os
import ssl
from loguru import logger
from pathlib import Path
import re

import nltk
import pandas as pd
from transformers import AutoTokenizer

from wattelse.common import BASE_DATA_PATH, BASE_OUTPUT_PATH, BASE_CACHE_PATH

# Ensures to write with +rw for both user and groups
os.umask(0o002)

# this is a workaround for downloading nltk data in some environments (https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords")

DATA_DIR = BASE_DATA_PATH / "bertopic"
OUTPUT_DIR = BASE_OUTPUT_PATH / "bertopic"
CACHE_DIR = BASE_CACHE_PATH / "bertopic"

TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
GROUPED_TIMESTAMP_COLUMN = "grouped_timestamp"
URL_COLUMN = "url"
TITLE_COLUMN = "title"
CITATION_COUNT_COL = "citation_count"

# Make dirs if not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(full_data_name: Path):
    logger.info(f"Loading data from: {full_data_name}")
    # Convert the Path object to a string before passing to file_to_pd
    df = file_to_pd(str(full_data_name), full_data_name.parent)
    # Convert timestamp column
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df.drop_duplicates(subset=["title"], keep="first")


def file_to_pd(file_name: str, base_dir: Path = None) -> pd.DataFrame:
    """Read data in various format and convert it to a DataFrame."""
    data_path = base_dir / file_name if base_dir else Path(file_name)
    data_path_str = str(data_path)  # Ensure the path is converted to string for pandas
    if ".csv" in file_name:
        return pd.read_csv(data_path_str)
    elif ".jsonl" in file_name or ".jsonlines" in file_name:
        return pd.read_json(data_path_str, lines=True)
    elif ".jsonl.gz" in file_name or ".jsonlines.gz" in file_name:
        with gzip.open(data_path_str, 'rt') as f_in:  # Open as text for JSON parsing
            return pd.read_json(f_in, lines=True)
    elif ".parquet" in file_name:
        return pd.read_parquet(data_path_str)



def clean_dataset(dataset: pd.DataFrame, length_criteria: int):
    """Clean dataset. So far, only removes short text."""
    cleaned_dataset = dataset.loc[
        dataset[TEXT_COLUMN].str.len() >= length_criteria
    ]
    return cleaned_dataset


def split_df_by_paragraphs(dataset: pd.DataFrame):
    """Split texts into multiple paragraphs and returns a concatenation of all extracts as a new pandas DF"""
    df = dataset.copy()  # to avoid modifying the original dataframe
    df[TEXT_COLUMN] = df[TEXT_COLUMN].str.split("\n")
    df = df.explode(TEXT_COLUMN)
    df = df[df[TEXT_COLUMN] != ""]
    return df


def split_df_by_paragraphs_v2(dataset: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512, min_length: int = 5):
    """Split text into multiple paragraphs all while ensuring that paragraphs aren't longer than embedding model's max sequence length"""
    df = dataset.copy()  # to avoid modifying the original dataframe
    new_rows = []
    for _, row in df.iterrows():
        doc = row['text']
        timestamp = row['timestamp']

        # Split the document into paragraphs
        paragraphs = re.split(r'\n+', doc)

        for paragraph in paragraphs:
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)

            current_doc = ""
            for sentence in sentences:
                sentence_ids = tokenizer.encode(sentence, padding=False, truncation=False, add_special_tokens=False)

                if len(sentence_ids) > max_length:
                    # If a single sentence is longer than max_length, split it into smaller chunks
                    sentence_chunks = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
                    for chunk in sentence_chunks:
                        new_row = row.copy()
                        new_row['text'] = chunk
                        new_row['is_split'] = True
                        new_row['num_tokens'] = len(tokenizer.encode(chunk, padding=False, truncation=False, add_special_tokens=False))
                        new_rows.append(new_row)
                else:
                    ids = tokenizer.encode(current_doc + " " + sentence, padding=False, truncation=False, add_special_tokens=False)
                    num_tokens = len(ids)

                    if num_tokens <= max_length:
                        current_doc += " " + sentence
                    else:
                        if current_doc.strip():
                            new_row = row.copy()
                            new_row['text'] = current_doc.strip()
                            new_row['is_split'] = True
                            new_row['num_tokens'] = len(tokenizer.encode(current_doc.strip(), padding=False, truncation=False, add_special_tokens=False))
                            new_rows.append(new_row)
                        current_doc = sentence

            if current_doc.strip():
                new_row = row.copy()
                new_row['text'] = current_doc.strip()
                new_row['is_split'] = True
                new_row['num_tokens'] = len(tokenizer.encode(current_doc.strip(), padding=False, truncation=False, add_special_tokens=False))
                new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    return new_df
