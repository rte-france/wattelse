#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import Optional, List, Dict

import chromadb
from chromadb import Embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from wattelse.api.embedding.client_embedding_api import EmbeddingAPI
from wattelse.common import BASE_DATA_DIR

DATABASE_PERSISTENCE_PATH = BASE_DATA_DIR / "rag_database"


class DataManagementError(Exception):
    pass


def get_collections():
    """
    Get all the collections in the database
    """
    persistent_client = chromadb.PersistentClient(str(DATABASE_PERSISTENCE_PATH))
    collections = persistent_client.list_collections()
    logger.debug(f"Collections: {[(col.name, col.id) for col in collections]}")
    return collections


def format_docs(docs: Document) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class DocumentCollection:
    def __init__(self, collection_name: str, embedding_function: Optional[Embeddings] = EmbeddingAPI()):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = chromadb.PersistentClient(
            str(DATABASE_PERSISTENCE_PATH)
        )
        self.collection = self.get_db_client()

    def get_db_client(self) -> Chroma:
        """
        Return a langchain Chroma client for the collection
        """
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"} # l2 is the default
        )
        return langchain_chroma

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add a list of texts to the collection in the database
        """
        ids = self.collection.add_texts(texts, metadata)
        logger.debug(f"Added {len(ids)} documents to the collection")

    def add_documents(self, documents: List[Document]):
        """
        Add a list of documents to the collection in the database
        """
        self.collection.add_documents(documents)

    def get_ids(self, file_name: str) -> List[str]:
        """Return all the chunks of the collection matching the file name passed as parameter"""
        data = self.collection.get(where={"file_name": file_name}, include=["metadatas"])
        return data["ids"]

    def is_present(self, file_name: str):
        """Return all file names in the database"""
        return len(self.get_ids(file_name)) > 0


def load_document_collection(group: str) -> DocumentCollection:
    """Retrieves the document collection for the given group name"""
    return DocumentCollection(group)
