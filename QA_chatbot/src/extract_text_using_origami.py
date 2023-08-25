from pathlib import Path

import pandas as pd
import typer
from loguru import logger

try:
    from origami_indexers.indexers import OrigamiSummaryDocXIndexer
    from origami_indexers.utils.s3 import download_file_if_not_exists
except ImportError as e:
    logger.error("Origami module is required for .docx upload feature")

DEFAULT_DATA_PATH = Path("./data")

DEFAULT_S3_PATH = "Etudes/2EDR/chatrelanglin/Documents de travail/1 - note d'etude/NT-CDI-NTS-SED Etude Explo Chatre langlin.docx"

def parse_docx(name: str = DEFAULT_S3_PATH, output_path: Path = DEFAULT_DATA_PATH) -> Path:
    logger.info(f"Parsing {name}...")

    local_fp = download_file_if_not_exists(
        name
    )

    output_file = local_fp.stem + ".csv"

    indexer = OrigamiSummaryDocXIndexer(local_fp,"")

    df = pd.DataFrame.from_dict([par[0] for par in indexer.paragraphs])
    full_path = output_path / output_file
    df.rename(columns={"content": "text", "pdf_page_number": "page_number"}).to_csv(full_path)
    logger.info(f"Saved data file: {full_path}")
    return full_path

if __name__ == "__main__":
    typer.run(parse_docx)