import configparser
import json
from typing import List

import requests
from pathlib import Path

from loguru import logger

from wattelse.api.rag_orchestrator import ENDPOINT_CHECK_SERVICE, ENDPOINT_CREATE_SESSION, ENDPOINT_QUERY_RAG, \
    ENDPOINT_UPLOAD_DOCS, ENDPOINT_SELECT_DOCS, ENDPOINT_REMOVE_DOCS, ENDPOINT_CURRENT_SESSIONS


class RAGAPIError(Exception):
    pass


class RAGOrchestratorClient:
    """Class in charge of routing requests to right backend depending on user group"""

    def __init__(self, login):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "rag_orchestrator.cfg")
        self.port = config.get("RAG_ORCHESTRATOR_API_CONFIG", "port")
        self.url = f'http://localhost:{self.port}'
        # one client is associated to one RAG session
        self.login = login
        if self.check_service():
            logger.debug("RAG Orchestrator is running")
            self.session_id = self.create_session()
        else:
            logger.error("Check RAG Orchestrator, does not seem to be running")

    def check_service(self) -> bool:
        """Check if RAG Orchestrator is running"""
        resp = requests.get(url=self.url + ENDPOINT_CHECK_SERVICE)
        return resp.json() == {"Status": "OK"}

    def create_session(self) -> str:
        """Create session associated to the current user"""
        response = requests.post(self.url + ENDPOINT_CREATE_SESSION, data=json.dumps({"login": self.login}))
        if response.status_code == 200:
            session_id = response.json()
            logger.info(f"Session id for user {self.login}: {session_id}")
            return session_id
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def upload_files(self, file_paths: List[Path]):
        files = [("files", open(p, "rb")) for p in file_paths]

        response = requests.post(url=f"{self.url}{ENDPOINT_UPLOAD_DOCS}/{self.session_id}", files=files)
        if response.status_code == 200:
            logger.info(response.json()["message"])
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def select_documents_by_name(self, doc_filenames: List[str] | None = None):
        """Select a subset of documents in the collection the user has access to, based on the provided titles.
        If the list of titles is empty or None, the full collection of documents is selected"""
        response = requests.post(url=f"{self.url}{ENDPOINT_SELECT_DOCS}/{self.session_id}",
                                 data=json.dumps(doc_filenames))
        if response.status_code == 200:
            logger.info(response.json()["message"])
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def select_documents_by_keywords(self, doc_titles: List[str] | None = None):
        # TODO: not implemented yet
        return "Not implemented yet"

    def remove_documents(self, doc_filenames: List[str]):
        """Removes documents from the collection the user has access to, as well as associated embeddings"""
        response = requests.post(url=f"{self.url}{ENDPOINT_REMOVE_DOCS}/{self.session_id}",
                                 data=json.dumps(doc_filenames))
        if response.status_code == 200:
            logger.info(response.json())
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def query_rag(self, query: str) -> str:
        """Query the RAG and returns an answer"""
        # TODO: handle memory parameter
        logger.debug(f"Question: {query}")
        response = requests.get(url=self.url + ENDPOINT_QUERY_RAG,
                                data=json.dumps({"query": query, "session_id": self.session_id}))
        if response.status_code == 200:
            logger.debug(f"Response: {response.json()}")
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def get_current_sessions(self) -> str:
        response = requests.get(url=self.url + ENDPOINT_CURRENT_SESSIONS)
        if response.status_code == 200:
            print(response.json())
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")


def main():
    docs = [
        "NMT - Avenant n°4 de révision à l'accord sur le temps de travail.pdf", "NMT - Accord télétravail.pdf"
    ]

    files = [Path(__file__).parent.parent.parent.parent / "data" / "chatbot" / "drh" / doc for doc in docs]

    # create one client per user
    rag_client = RAGOrchestratorClient("alice")

    # upload data files
    rag_client.upload_files(files)

    # select document selection
    rag_client.select_documents_by_name([docs[1]])

    # query rag based on selected document collection
    rag_client.query_rag("Quoi de neuf, doc?")
    rag_client.query_rag("Y'a une prime pour le télétravail?")

    # delete doc
    rag_client.remove_documents([docs[0]])

    # current sessions
    rag_client.get_current_sessions()

if __name__ == "__main__":
    main()
