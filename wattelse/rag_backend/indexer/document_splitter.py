#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    TextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLSectionSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_index.core import node_parser

from wattelse.rag_backend.indexer import CODE_EXTENSIONS, CFG_EXTENSIONS
from wattelse.rag_backend.indexer.structured_document_header_hierarchy import (
    get_html_hierarchy,
    get_markdown_hierarchy,
)
import re


class SentenceSplitter(TextSplitter):
    """This class extends langchain TextSplitter in order to split documents/texts by sentence."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # split text per sentence using llama_index
        sentence_parser = node_parser.SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
        )
        text_chunks = sentence_parser.split_text(text)
        return text_chunks


def split_file(
    file_extension: str, docs: List[Document], use_sentence_splitter: bool = True
) -> List[Document]:
    """Split a file into smaller chunks - the chunking method depends on file type"""
    if file_extension == ".md":
        text_splitter = MarkdownHeaderTextSplitter(
            return_each_line=False,
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
        )
        splits = []
        for doc in docs:
            new_docs = text_splitter.split_text(doc.page_content)
            new_docs = get_markdown_hierarchy(
                new_docs
            )  # concatenates parent-headers to the beginning of each subsection
            for d in new_docs:
                d.metadata = doc.metadata
            splits += new_docs
        return splits
    elif file_extension in [".htm", ".html"]:
        text_splitter = HTMLSectionSplitter(
            return_each_element=False,
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ],
        )  # NB. HTMLHeaderTextSplitter replaced with HTMLSectionSplitter
        splits = []
        for doc in docs:
            new_docs = text_splitter.split_text(doc.page_content)
            new_docs = get_html_hierarchy(
                new_docs
            )  # concatenates parent-headers to the beginning of each subsection
            for d in new_docs:
                d.page_content = re.sub(
                    r"[^\S\n]{2,}", " ", d.page_content
                )  # removes extra spaces that are not line returns
                d.metadata = doc.metadata
            splits += new_docs
        return splits
    elif file_extension in [".csv", ".xlsx"] + CFG_EXTENSIONS + CODE_EXTENSIONS:
        # already split
        return docs
    else:
        if use_sentence_splitter:
            text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)
        else:  # TODO: check split parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
        splits = text_splitter.split_documents(docs)
        return splits
