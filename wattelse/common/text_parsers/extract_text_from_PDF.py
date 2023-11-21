import re
from pathlib import Path

import fitz  # the package is called pymupdf (confusing!!)
import pandas as pd
import typer
from loguru import logger

DEFAULT_DATA_PATH = Path("../../../data/chatbot")

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


def extract_pages_from_pdf(pdf_file_path, filter_value = 10) -> pd.DataFrame:
    """Extracts text from PDF. Each extract correspond to the content of 1 PDF page.

    Args:
        pdf_file_path (string): path to the pdf to parse.
        filter_value (int): pages with less than filter_value words will be removed.

    Returns:
        pd.DataFrame: DataFrame having columns "text" and "page_number".
    """
    data_dict = {"text": [], "page_number": []}

    ### Load pdf and parse pages of text
    with fitz.open(pdf_file_path) as f:
        for page in f:
            text = page.get_text()
            data_dict["text"].append(text)  # store text only
            data_dict["page_number"].append(page.number)

    ### Process collected pages
    # Transform into a pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict)
    logger.info(f"Found {len(df)} pages")

    # Clean text
    df["text"] = df["text"].apply(_clean_text).astype(str)
    
    # Filter blocks to keep paragraphs only
    logger.debug(f"Removing extracts having less than {filter_value} words...")
    df = df[df["text"].str.split().apply(len) > filter_value].reset_index(drop=True)
    logger.debug(f"{len(df)} pages remaining")
    return df

def extract_chunks_from_pdf(pdf_file_path, chunk_size = 100, chunk_overlap = 20) -> pd.DataFrame:
    """Extracts text from PDF. Each extract correspond to the content of 1 PDF page.
    
    Args:
        pdf_file_path (string): path to the pdf to parse.
        chunk_size (int): chunk size.
        chunk_overlap (int): chunk overlap (overlaping both previous and next chunks).

    Returns:
        pd.DataFrame: DataFrame having columns "text".
    """
    ### Load pdf and get full text
    full_text = ""
    with fitz.open(pdf_file_path) as f:
        for page in f:
            full_text += page.get_text()
    
    full_text = _clean_text(full_text) # clean text
    full_text = full_text.split(" ")

    text_chunks = []
    for i in range((len(full_text)-chunk_size+chunk_overlap)//(chunk_size-chunk_overlap)):
        text_chunks.append(" ".join(full_text[i*chunk_size-i*chunk_overlap:(i+1)*chunk_size-i*chunk_overlap]))
    text_chunks.append(full_text[-chunk_size:]) # add last chunk

    # Transform into a pandas DataFrame
    df = pd.DataFrame.from_dict({"text": text_chunks})
    logger.info(f"Extracted {len(df)} chunks")

    return df

def parse_pdf(pdf_file: Path, output_path: Path = DEFAULT_DATA_PATH, mode: str = "chunk") -> Path:
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

    if mode=="chunk":
        df = extract_chunks_from_pdf(pdf_file)
    elif mode=="page":
        df = extract_pages_from_pdf(pdf_file)

    df.to_csv(full_output_path)
    logger.info(f"Saved data file: {full_output_path}")

    return full_output_path

if __name__ == "__main__":
    typer.run(parse_pdf)