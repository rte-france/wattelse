#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os
import re
from typing import List, Iterator

import bs4
import pandas as pd
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    TextLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from wattelse.indexer import CODE_EXTENSIONS, CFG_EXTENSIONS


class ParsingException(Exception):
    pass


def parse_file(file: Path) -> List[Document]:
    extension = file.suffix.lower()
    if extension in [".txt"] + CFG_EXTENSIONS + [
        ".cc",
        ".h",
    ]:  # NB .cc, .h handled as text files due to current limitation of langchain LanguageParser
        docs = _parse_txt(file)
    elif extension == ".pdf":
        docs = _parse_pdf(file)
    elif extension == ".docx":
        docs = _parse_docx(file)
    elif extension == ".pptx":
        docs = _parse_pptx(file)
    elif extension == ".xlsx":
        docs = _parse_xslx(file)
    elif extension == ".md":
        docs = _parse_md(file)
    elif extension == ".csv":
        docs = _parse_csv(file)
    elif extension in [".htm", ".html"]:
        docs = _parse_html(file)
    elif extension in CODE_EXTENSIONS:
        docs = _parse_language(file, extension)
    else:
        raise ParsingException(f"Unsupported file format: {file.suffix}!")

    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
        # Add additional metadata
        doc.metadata["file_name"] = file.name

    return docs


def parse_url(url: str) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        proxies={
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        },
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            ),
        ),
    )
    docs = loader.load()
    return docs


def _parse_txt(file: Path) -> List[Document]:
    loader = TextLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document per page
    return data


def _parse_pdf(file: Path) -> List[Document]:
    loader = PyMuPDFLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document per page
    return data


def _parse_docx(file: Path) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document
    return data


def _parse_pptx(file: Path) -> List[Document]:
    loader = UnstructuredPowerPointLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document
    return data


def _parse_xslx(file: Path) -> List[Document]:
    loader = CustomXLSXDocumentLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document per line
    return data


def _parse_md(file: Path) -> List[Document]:
    loader = TextLoader(
        file.absolute().as_posix()
    )  # NB. UnstructuredMarkdownLoader removes markdown structure tags
    data = loader.load()
    # NB. return one document
    return data


def _parse_csv(file: Path) -> List[Document]:
    loader = CSVLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document per row
    return data


def _parse_html(file: Path) -> List[Document]:
    loader = TextLoader(
        file.absolute().as_posix()
    )  # NB. BSHTMLLoader removes html tags
    data = loader.load()
    # NB. return one document
    return data


def _parse_language(file: Path, extension: str) -> List[Document]:
    # Each top-level function and class in the code is loaded into separate documents. Furthermore, an extra document
    # is generated, containing the remaining top-level code that excludes the already segmented functions and classes.
    blob = Blob.from_path(file)
    parser = LanguageParser(language=extension[1:])
    return parser.parse(blob)


def _clean_text(x):
    """
    Function applied to clean the text
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


class CustomXLSXDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path
        dfs = pd.read_excel(self.file_path, sheet_name=None)  # read all excel sheets
        dfs = {
            sheet: df.ffill().astype(
                str
            )  # ffill used to resolve merged cells (duplicate content)
            for sheet, df in dfs.items()
        }
        self.dfs = dfs

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader that reads a file line by line.
        Returns a generator to yield documents one by one.
        """
        for sheet, df in self.dfs.items():
            lines = df.to_dict(orient="records")
            for line in lines:
                line = {key: value for key, value in line.items() if value != "nan"}
                if len(self.dfs) > 1:  # add sheet in info if multiple sheet
                    line["sheet"] = sheet
                yield Document(
                    page_content=str(line),
                    metadata={
                        "source": self.file_path,
                        "sheet": sheet,
                    },
                )
