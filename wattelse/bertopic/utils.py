#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import gzip
import os
import ssl
from loguru import logger
from pathlib import Path

import nltk
import pandas as pd

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


def load_data(full_data_name: str):
    logger.info(f"Loading data from: {full_data_name}")
    df = file_to_pd(full_data_name)
    # convert timestamp column
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df.drop_duplicates(
                subset=["title"], keep="first", inplace=False
            )


def file_to_pd(file_name: str, base_dir: Path = None) -> pd.DataFrame:
    """Read data in various format and convert in to a DataFrame"""
    data_path = base_dir / file_name if base_dir else Path(file_name)
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


def split_df_by_paragraphs(dataset: pd.DataFrame):
    """Split texts into multiple paragraphs and returns a concatenation of all extracts as a new pandas DF"""
    df = dataset.copy()  # to avoid modifying the original dataframe
    df[TEXT_COLUMN] = df[TEXT_COLUMN].str.split("\n")
    df = df.explode(TEXT_COLUMN)
    df = df[df[TEXT_COLUMN] != ""].reset_index(drop=True)
    return df

