from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from wattelse.common import TEXT_COLUMN, FILENAME_COLUMN
from wattelse.common.vars import BASE_DATA_DIR

try:
    from origami_indexers.indexers import OrigamiSummaryPdfIndexer, OrigamiSummaryDocXIndexer
except ImportError as e:
    logger.error("Origami module is required for .docx upload feature")

#TODO: ideally, the call to the Origami indexer shall be done using a API call to avoid conflicts of versions, etc.

def parse_docx(name: Path, output_path: Path = BASE_DATA_DIR) -> Path:
    logger.info(f"Parsing {name}...")

    output_file = name.stem + ".csv"

    if name.suffix.lower() == ".docx":
        indexer = OrigamiSummaryDocXIndexer(name,"")
    elif name.suffix.lower() == ".pdf":
        indexer = OrigamiSummaryPdfIndexer(name,"")
    else:
        logger.error("Format not supported")
        return -1

    df = pd.DataFrame.from_dict([par[0] for par in indexer.paragraphs])
    full_path = output_path / output_file
    df[FILENAME_COLUMN] = name
    df.rename(columns={"content": TEXT_COLUMN, "pdf_page_number": "page_number"}).to_csv(full_path)
    logger.info(f"Saved data file: {full_path}")
    return full_path

if __name__ == "__main__":
    typer.run(parse_docx)