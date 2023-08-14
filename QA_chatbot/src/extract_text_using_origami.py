from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from origami_indexers.indexers import OrigamiSummaryDocXIndexer
from origami_indexers.utils.s3 import download_file_if_not_exists

DEFAULT_DATA_PATH = Path("./data")

DEFAULT_S3_PATH = "Etudes/2EDR/chatrelanglin/Documents de travail/1 - note d'etude/NT-CDI-NTS-SED Etude Explo Chatre langlin.docx"

def parse_document(name: str = DEFAULT_S3_PATH, output_path: Path = DEFAULT_DATA_PATH):
    logger.info(f"Parsing {name}...")

    local_fp = download_file_if_not_exists(
        name
    )

    output_file = local_fp.stem + ".csv"

    indexer = OrigamiSummaryDocXIndexer(local_fp,"")

    df = pd.DataFrame.from_dict([par[0] for par in indexer.paragraphs])
    df.rename(columns={"content": "text", "pdf_page_number": "page_number"}).to_csv(output_path / output_file)

if __name__ == "__main__":
    typer.run(parse_document)