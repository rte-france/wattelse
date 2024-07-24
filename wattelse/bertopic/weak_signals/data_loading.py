import pandas as pd
import os
import glob
import re
import json
from typing import Dict
import streamlit as st
from typing import Tuple
from global_vars import DATA_PATH
from wattelse.bertopic.utils import (
    preprocess_french_text,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    URL_COLUMN,
)

# @st.cache_data
def load_and_preprocess_data(selected_file: Tuple[str, str], language: str, min_chars: int, split_by_paragraph: bool) -> pd.DataFrame:
    """
    Load and preprocess data from a selected file.
    
    Args:
        selected_file (tuple): A tuple containing the selected file name and extension.
        language (str): The language of the text data ('French' or 'English').
        min_chars (int): The minimum number of characters required for a text to be included.
        split_by_paragraph (bool): Whether to split the text data by paragraphs.
    
    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    file_name, file_ext = selected_file
    
    if file_ext == 'csv':
        df = pd.read_csv(os.path.join(DATA_PATH, file_name))
    elif file_ext == 'parquet':
        df = pd.read_parquet(os.path.join(DATA_PATH, file_name))
    elif file_ext == 'json':
        df = pd.read_json(os.path.join(DATA_PATH, file_name))
    elif file_ext == 'jsonl':
        df = pd.read_json(os.path.join(DATA_PATH, file_name), lines=True)
    
    # Convert timestamp column to datetime
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors='coerce')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=[TIMESTAMP_COLUMN])
    
    df = df.sort_values(by=TIMESTAMP_COLUMN, ascending=True).reset_index(drop=True)
    df['document_id'] = df.index
    
    if URL_COLUMN in df.columns:
        df['source'] = df[URL_COLUMN].apply(lambda x: x.split('/')[2] if pd.notna(x) else None)
    else:
        df['source'] = None
        df[URL_COLUMN] = None
    
    if language == "French":
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_french_text)
    
    if split_by_paragraph:
        new_rows = []
        for _, row in df.iterrows():
            # Attempt splitting by \n\n first
            paragraphs = re.split(r'\n\n', row[TEXT_COLUMN])
            if len(paragraphs) == 1:  # If no split occurred, attempt splitting by \n
                paragraphs = re.split(r'\n', row[TEXT_COLUMN])
            for paragraph in paragraphs:
                new_row = row.copy()
                new_row[TEXT_COLUMN] = paragraph
                new_row['source'] = row['source']
                new_rows.append(new_row)
        df = pd.DataFrame(new_rows)
    
    if min_chars > 0:
        df = df[df[TEXT_COLUMN].str.len() >= min_chars]
    
    df = df[df[TEXT_COLUMN].str.strip() != ''].reset_index(drop=True)
    
    return df

def group_by_days(df: pd.DataFrame, day_granularity: int = 1) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Group a DataFrame by a specified number of days.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing a TIMESTAMP_COLUMN column.
        day_granularity (int): The number of days to group by (default is 1).
    
    Returns:
        Dict[pd.Timestamp, pd.DataFrame]: A dictionary where each key is the timestamp group and the value is the corresponding DataFrame.
    """
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    grouped = df.groupby(pd.Grouper(key=TIMESTAMP_COLUMN, freq=f'{day_granularity}D'))
    dict_of_dfs = {name: group for name, group in grouped}
    return dict_of_dfs