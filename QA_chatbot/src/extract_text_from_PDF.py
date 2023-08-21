import re
from pathlib import Path

import fitz  # the package is called pymupdf (confusing!!)
import pandas as pd
import typer
from loguru import logger

DEFAULT_DATA_PATH = Path("./data")
FILTER_TEXT_VALUE = 10 # blocks with less than filter_text_value words will be discarded


def _clean_text(x):
    """
    Function applied to clean the text column in the DataFrame. 
    """
    
    # Weird behavior of pymupdf
    x = re.sub("ff ", "ff", x)
    x = re.sub("ﬁ ", "fi", x)
    x = re.sub("fi ", "fi", x)
    x = re.sub("�", " ", x)
    
    # Remove block structure
    x = re.sub("-\n", "", x)
    x = re.sub("\n", " ", x)
    x = re.sub(" +", " ", x)
    x = x.strip()

    return x


def extract_data_from_pdf(pdf_file_path) -> pd.DataFrame:
    data_dict = {"text": [], "page_number": []}

    ### Load pdf and parse blocks of text
    with fitz.open(pdf_file_path) as f:
        for page in f:
            text = page.get_text("blocks")
            for elem in text:
                data_dict["text"].append(elem[4])  # store text only
                data_dict["page_number"].append(page.number)

    ### Process collected blocks
    # Transform into a pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict)
    logger.info(f"Found {len(df)} blocks of text")

    # Clean text
    df["text"] = df["text"].apply(_clean_text).astype(str)
    
    # Filter blocks to keep paragraphs only
    logger.debug(f"Removing blocks having less than {FILTER_TEXT_VALUE} words...")
    df = df[df["text"].str.split().apply(len) > FILTER_TEXT_VALUE].reset_index(drop=True)
    logger.debug(f"{len(df)} blocks remaining")
    return df

def parse_pdf(pdf_file: Path, output_path: Path = DEFAULT_DATA_PATH) -> Path:
    logger.info(f"Parsing {pdf_file}...")

    output_file = pdf_file.stem + ".csv"
    full_output_path = output_path / output_file

    df = extract_data_from_pdf(pdf_file)

    df.to_csv(full_output_path)
    logger.info(f"Saved data file: {full_output_path}")

    return full_output_path

if __name__ == "__main__":
    typer.run(parse_pdf)
        