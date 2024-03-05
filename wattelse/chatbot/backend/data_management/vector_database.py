from datetime import datetime
from typing import Optional, List, Dict

import chromadb
from chromadb import Embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from wattelse.api.embedding.client_embedding_api import EmbeddingAPI
from wattelse.chatbot.backend.user_management.user_manager import get_all_document_collections
from wattelse.common import BASE_DATA_DIR

DATABASE_PERSISTENCE_PATH = BASE_DATA_DIR / "rag_database.db"


def initialize_database():
    """
    Initialize the database
    """
    persistent_client = chromadb.PersistentClient(str(DATABASE_PERSISTENCE_PATH))
    for collection_id in get_all_document_collections():
        persistent_client.create_collection(name=collection_id, metadata={"created_at": str(datetime.utcnow())})
    logger.debug(f"Database initialized: {DATABASE_PERSISTENCE_PATH}")


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
        # self.collection and self.db_client are two ways to access the DB,
        # self.collection using native Chroma DB functions,
        # self.db_client using the langchain wrapper
        # Depending on the usage, one may be more convenient than the other
        #self.collection = self.client.get_or_create_collection(name=collection_name,
        #                                                       embedding_function=embedding_function)
        self.collection = self.get_db_client()


    def get_db_client(self) -> Chroma:
        """
        Return a langchain Chroma client for the collection
        """
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )
        return langchain_chroma

    def delete_documents(self, documents_titles: List[str]):
        """
        Delete all documents specified in `documents_titles` from the collection specified by `collection_name`
        """
        # TODO: à implémenter
        return

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add a list of texts to the collection in the database
        """
        ids = self.collection.add_texts(texts, metadatas)
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

