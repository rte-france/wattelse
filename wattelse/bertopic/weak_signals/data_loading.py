import pandas as pd
import os
import glob
import re
import json
from typing import Dict
import streamlit as st
from typing import Tuple
from global_vars import DATA_PATH

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
    text = re.sub(r'\b(-|/|;|:)', ' ', text)
    
    # Replace special characters with a space (preserving specified punctuation, accented characters, common Latin extensions, and newlines)
    text = re.sub(r'[^\w\s\nàâçéèêëîïôûùüÿñæœ.,\'\"\(\)\[\]]', ' ', text)
    
    # Normalize superscripts and subscripts for both numbers and letters
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻ",
                                    "0123456789abcdefghijklmnopqrstuvwxyz")
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑᵢₒᵣᵤᵥₓ",
                                  "0123456789aeioruvx")
    text = text.translate(superscript_map)
    text = text.translate(subscript_map)

    # Split words that contain capitals in the middle but avoid splitting fully capitalized words
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text



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
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    df['document_id'] = df.index
    
    if 'url' in df.columns:
        df['source'] = df['url'].apply(lambda x: x.split('/')[2] if pd.notna(x) else None)
    else:
        df['source'] = None
        df['url'] = None
    
    if language == "French":
        df['text'] = df['text'].apply(preprocess_french_text)
    
    if split_by_paragraph:
        new_rows = []
        for _, row in df.iterrows():
            # Attempt splitting by \n\n first
            paragraphs = re.split(r'\n\n', row['text'])
            if len(paragraphs) == 1:  # If no split occurred, attempt splitting by \n
                paragraphs = re.split(r'\n', row['text'])
            for paragraph in paragraphs:
                new_row = row.copy()
                new_row['text'] = paragraph
                new_row['source'] = row['source']
                new_rows.append(new_row)
        df = pd.DataFrame(new_rows)
    
    if min_chars > 0:
        df = df[df['text'].str.len() >= min_chars]
    
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)
    
    return df

def group_by_days(df: pd.DataFrame, day_granularity: int = 1) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Group a DataFrame by a specified number of days.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'timestamp' column.
        day_granularity (int): The number of days to group by (default is 1).
    
    Returns:
        Dict[pd.Timestamp, pd.DataFrame]: A dictionary where each key is the timestamp group and the value is the corresponding DataFrame.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    grouped = df.groupby(pd.Grouper(key='timestamp', freq=f'{day_granularity}D'))
    dict_of_dfs = {name: group for name, group in grouped}
    return dict_of_dfs