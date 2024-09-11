#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import gzip
import os
import ssl
import re
from pathlib import Path

import nltk
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger

from wattelse.common import BASE_DATA_PATH, BASE_OUTPUT_PATH, BASE_CACHE_PATH

# Ensures files are written with +rw permissions for both user and groups
os.umask(0o002)

# Workaround for downloading nltk data in some environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords")

# Define directories
DATA_DIR = BASE_DATA_PATH / "bertopic"
OUTPUT_DIR = BASE_OUTPUT_PATH / "bertopic"
CACHE_DIR = BASE_CACHE_PATH / "bertopic"

# Define column names
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
GROUPED_TIMESTAMP_COLUMN = "grouped_timestamp"
URL_COLUMN = "url"
TITLE_COLUMN = "title"
CITATION_COUNT_COL = "citation_count"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLOTLY_BUTTON_SAVE_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        # 'height': 500,
        # 'width': 1500,
        "scale": 1,
    }
}


def load_data(full_data_name: Path) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Args:
        full_data_name (Path): The path to the data file.

    Returns:
        pd.DataFrame: Loaded data with timestamp column converted to datetime.
    """
    logger.info(f"Loading data from: {full_data_name}")
    df = file_to_pd(str(full_data_name), full_data_name.parent)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df.drop_duplicates(subset=["title"], keep="first")


def file_to_pd(file_name: str, base_dir: Path = None) -> pd.DataFrame:
    """
    Read data in various formats and convert it to a DataFrame.

    Args:
        file_name (str): The name of the file to read.
        base_dir (Path, optional): The base directory of the file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    data_path = base_dir / file_name if base_dir else Path(file_name)
    data_path_str = str(data_path)

    if file_name.endswith(".csv"):
        return pd.read_csv(data_path_str)
    elif file_name.endswith(".jsonl") or file_name.endswith(".jsonlines"):
        return pd.read_json(data_path_str, lines=True)
    elif file_name.endswith(".jsonl.gz") or file_name.endswith(".jsonlines.gz"):
        with gzip.open(data_path_str, "rt") as f_in:
            return pd.read_json(f_in, lines=True)
    elif file_name.endswith(".parquet"):
        return pd.read_parquet(data_path_str)


def clean_dataset(dataset: pd.DataFrame, length_criteria: int) -> pd.DataFrame:
    """
    Clean the dataset by removing short texts.

    Args:
        dataset (pd.DataFrame): The dataset to clean.
        length_criteria (int): The minimum length of text to keep.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    cleaned_dataset = dataset.loc[dataset[TEXT_COLUMN].str.len() >= length_criteria]
    return cleaned_dataset


def split_df_by_paragraphs(
    dataset: pd.DataFrame,
    enhanced: bool = False,
    tokenizer: AutoTokenizer = None,
    max_length: int = 512,
    min_length: int = 5,
) -> pd.DataFrame:
    """
    Split texts into multiple paragraphs and return a concatenation of all extracts as a new DataFrame.
    If enhanced is True, it also considers the embedding model's max sequence length.

    Args:
        dataset (pd.DataFrame): The dataset to split.
        enhanced (bool): Whether to use the enhanced splitting method.
        tokenizer (AutoTokenizer): The tokenizer to use for splitting (required if enhanced is True).
        max_length (int): The maximum length of tokens (used if enhanced is True).
        min_length (int): The minimum length of tokens (used if enhanced is True).

    Returns:
        pd.DataFrame: The dataset with texts split into paragraphs.
    """
    if not enhanced:
        df = dataset.copy()
        df[TEXT_COLUMN] = df[TEXT_COLUMN].str.split("\n")
        df = df.explode(TEXT_COLUMN)
        df = df[df[TEXT_COLUMN] != ""]
        return df

    if tokenizer is None:
        raise ValueError("Tokenizer is required for enhanced splitting.")

    df = dataset.copy()
    new_rows = []

    for _, row in df.iterrows():
        doc = row[TEXT_COLUMN]
        timestamp = row[TIMESTAMP_COLUMN]

        paragraphs = re.split(r"\n+", doc)

        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current_doc = ""
            for sentence in sentences:
                sentence_ids = tokenizer.encode(
                    sentence, padding=False, truncation=False, add_special_tokens=False
                )

                if len(sentence_ids) > max_length:
                    sentence_chunks = [
                        sentence[i : i + max_length]
                        for i in range(0, len(sentence), max_length)
                    ]
                    for chunk in sentence_chunks:
                        new_row = row.copy()
                        new_row[TEXT_COLUMN] = chunk
                        new_row["is_split"] = True
                        new_row["num_tokens"] = len(
                            tokenizer.encode(
                                chunk,
                                padding=False,
                                truncation=False,
                                add_special_tokens=False,
                            )
                        )
                        new_rows.append(new_row)
                else:
                    ids = tokenizer.encode(
                        current_doc + " " + sentence,
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                    )
                    num_tokens = len(ids)

                    if num_tokens <= max_length:
                        current_doc += " " + sentence
                    else:
                        if current_doc.strip():
                            new_row = row.copy()
                            new_row[TEXT_COLUMN] = current_doc.strip()
                            new_row["is_split"] = True
                            new_row["num_tokens"] = len(
                                tokenizer.encode(
                                    current_doc.strip(),
                                    padding=False,
                                    truncation=False,
                                    add_special_tokens=False,
                                )
                            )
                            new_rows.append(new_row)
                        current_doc = sentence

            if current_doc.strip():
                new_row = row.copy()
                new_row[TEXT_COLUMN] = current_doc.strip()
                new_row["is_split"] = True
                new_row["num_tokens"] = len(
                    tokenizer.encode(
                        current_doc.strip(),
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                    )
                )
                new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def preprocess_french_text(text: str) -> str:
    """
    Preprocess French text by normalizing apostrophes, replacing hyphens and similar characters with spaces,
    removing specific prefixes, removing unwanted punctuations (excluding apostrophes, periods, commas, and specific other punctuation),
    replacing special characters with a space (preserving accented characters, common Latin extensions, and newlines),
    normalizing superscripts and subscripts,
    splitting words containing capitals in the middle (while avoiding splitting fully capitalized words),
    and replacing multiple spaces with a single space.

    Args:
        text (str): The input French text to preprocess.

    Returns:
        str: The preprocessed French text.
    """
    # Normalize different apostrophe variations to a standard apostrophe
    text = text.replace("’", "'")

    # Replace hyphens and similar characters with spaces
    text = re.sub(r"\b(-|/|;|:)", " ", text)

    # Replace special characters with a space (preserving specified punctuation, accented characters, common Latin extensions, and newlines)
    text = re.sub(r"[^\w\s\nàâçéèêëîïôûùüÿñæœ.,\'\"\(\)\[\]]", " ", text)

    # Normalize superscripts and subscripts for both numbers and letters
    superscript_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻ", "0123456789abcdefghijklmnopqrstuvwxyz"
    )
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑᵢₒᵣᵤᵥₓ", "0123456789aeioruvx")
    text = text.translate(superscript_map)
    text = text.translate(subscript_map)

    # Split words that contain capitals in the middle but avoid splitting fully capitalized words
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    return text
