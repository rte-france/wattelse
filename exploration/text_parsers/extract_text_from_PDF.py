#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import re
from pathlib import Path

import fitz  # the package is called pymupdf (confusing!!)
import pandas as pd
import typer
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from wattelse import TEXT_COLUMN, FILENAME_COLUMN, BASE_DATA_PATH


def _clean_text(x):
    """
    Function applied to clean the text column in the DataFrame.
    """

    # Weird behavior of pymupdf
    x = re.sub("ff ", "ff", x)
    x = re.sub("ﬁ ", "fi", x)
    x = re.sub("fi ", "fi", x)
    x = re.sub("�", " ", x)

    # Remove PDF structure
    x = re.sub("-\n", "", x)
    x = re.sub(r"(?<!\n)\n(?!\n)", " ", x)
    x = re.sub(" +", " ", x)
    x = x.strip()

    return x


def extract_pages_from_pdf(pdf_file_path, filter_value=10) -> pd.DataFrame:
    """Extracts text from PDF. Each extract correspond to the content of 1 PDF page.

    Args:
        pdf_file_path (string): path to the pdf to parse.
        filter_value (int): pages with less than filter_value words will be removed.

    Returns:
        pd.DataFrame: DataFrame having columns TEXT_COLUMN and "page_number".
    """
    data_dict = {TEXT_COLUMN: [], "page_number": []}

    ### Load pdf and parse pages of text
    with fitz.open(pdf_file_path) as f:
        for page in f:
            text = page.get_text()
            data_dict[TEXT_COLUMN].append(text)  # store text only
            data_dict["page_number"].append(page.number)

    ### Process collected pages
    # Transform into a pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict)
    logger.info(f"Found {len(df)} pages")

    # Clean text
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(_clean_text).astype(str)

    # Filter blocks to keep paragraphs only
    logger.debug(f"Removing extracts having less than {filter_value} words...")
    df = df[df[TEXT_COLUMN].str.split().apply(len) > filter_value].reset_index(
        drop=True
    )
    logger.debug(f"{len(df)} pages remaining")
    return df


def extract_chunks_from_pdf(
    pdf_file_path, chunk_size=200, chunk_overlap=50
) -> pd.DataFrame:
    """Extracts text from PDF. Each extract correspond to a chunk with some overlap with nearby other chunks.

    Args:
        pdf_file_path (string): path to the pdf to parse.
        chunk_size (int): chunk size.
        chunk_overlap (int): chunk overlap (overlaping both previous and next chunks).

    Returns:
        pd.DataFrame: DataFrame having columns TEXT_COLUMN.
    """
    ### Load pdf and get full text
    full_text = ""
    with fitz.open(pdf_file_path) as f:
        for page in f:
            full_text += page.get_text()

    full_text = _clean_text(full_text)  # clean text

    # split text per sentence using llama_index
    sentence_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n"
    )
    text_chunks = sentence_parser.split_text(full_text)

    # Transform into a pandas DataFrame
    df = pd.DataFrame.from_dict({TEXT_COLUMN: text_chunks})
    logger.info(f"Extracted {len(df)} chunks")

    return df


def parse_pdf(
    pdf_file: Path, output_path: Path = BASE_DATA_PATH, mode: str = "chunk"
) -> Path:
    """Parse a pdf file using defined mode.

    Args:
        pdf_file (Path): path to the pdf to parse.
        output_path (Path): output path.
        mode (str):
                    - `page` : use extract_pages_from_pdf
                    - `chunk` : use extract_chunks_from_pdf
    """
    logger.info(f"Parsing {pdf_file}...")

    output_file = pdf_file.stem + ".csv"
    full_output_path = output_path / output_file

    if mode == "chunk":
        df = extract_chunks_from_pdf(pdf_file)
    elif mode == "page":
        df = extract_pages_from_pdf(pdf_file)

    df[FILENAME_COLUMN] = pdf_file
    df.to_csv(full_output_path)
    logger.info(f"Saved data file: {full_output_path}")

    return full_output_path


if __name__ == "__main__":
    typer.run(parse_pdf)
