#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, \
    RecursiveCharacterTextSplitter
from llama_index.core import node_parser


class SentenceSplitter(TextSplitter):
    """This class extends langchain TextSplitter in order to split documents/texts by sentence."""
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # split text per sentence using llama_index
        sentence_parser = node_parser.SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                                       paragraph_separator="\n\n")
        text_chunks = sentence_parser.split_text(text)
        return text_chunks


def split_file(file_extension: str, docs: List[Document], use_sentence_splitter: bool = True) -> List[Document]:
    """Split a file into smaller chunks - the chunking method depends on file type"""
    if file_extension == ".md":
        text_splitter = MarkdownHeaderTextSplitter(return_each_line=True, headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        splits = []
        for doc in docs:
            new_docs = text_splitter.split_text(doc.page_content)
            for d in new_docs:
                d.metadata = doc.metadata
            splits += new_docs
        return splits
    elif file_extension in [".htm", ".html"]:
        text_splitter = HTMLHeaderTextSplitter(return_each_element=True, headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ])
        splits = []
        for doc in docs:
            new_docs = text_splitter.split_text(doc.page_content)
            for d in new_docs:
                d.metadata = doc.metadata
            splits += new_docs
        return splits
    elif file_extension == ".csv" or file_extension == ".xlsx":
        # already split by row
        return docs
    else:
        if use_sentence_splitter:
            text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)
        else:   # TODO: check split parameters
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        return splits
