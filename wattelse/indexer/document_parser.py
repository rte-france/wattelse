import os
import re
from typing import List

import bs4
from langchain_community.document_loaders import PyMuPDFLoader, BSHTMLLoader, UnstructuredPowerPointLoader, \
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pathlib import Path

from langchain_core.documents import Document


class ParsingException(Exception):
    pass


def parse_file(file: Path) -> List[Document]:
    extension = file.suffix.lower()
    if extension == ".pdf":
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
    else:
        raise ParsingException(f"Unsupported file format: {file.suffix}!")

    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
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
    loader = UnstructuredExcelLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document
    return data


def _parse_md(file: Path) -> List[Document]:
    loader = UnstructuredMarkdownLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document
    return data


def _parse_csv(file: Path) -> List[Document]:
    loader = CSVLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document per row
    return data


def _parse_html(file: Path) -> List[Document]:
    loader = BSHTMLLoader(file.absolute().as_posix())
    data = loader.load()
    # NB. return one document
    return data


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
