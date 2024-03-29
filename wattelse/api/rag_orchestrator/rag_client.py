#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
import json
from typing import List, Dict

import requests
from pathlib import Path

from loguru import logger

from wattelse.api.rag_orchestrator import ENDPOINT_CHECK_SERVICE, ENDPOINT_CREATE_SESSION, ENDPOINT_QUERY_RAG, \
    ENDPOINT_UPLOAD_DOCS, ENDPOINT_SELECT_DOCS, ENDPOINT_REMOVE_DOCS, ENDPOINT_CURRENT_SESSIONS, \
    ENDPOINT_SELECT_BY_KEYWORDS, ENDPOINT_LIST_AVAILABLE_DOCS


class RAGAPIError(Exception):
    pass


class RAGOrchestratorClient:
    """Class in charge of routing requests to right backend depending on user group"""

    def __init__(self, login: str, group: str):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "rag_orchestrator.cfg")
        self.port = config.get("RAG_ORCHESTRATOR_API_CONFIG", "port")
        self.url = f'http://localhost:{self.port}'
        # one client is associated to one RAG session
        self.login = login
        self.group = group
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
        response = requests.post(self.url + ENDPOINT_CREATE_SESSION, data=json.dumps({"login": self.login, "group": self.group}))
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
        If the list of titles is empty or None, the full collection of documents is selected for the RAG"""
        response = requests.post(url=f"{self.url}{ENDPOINT_SELECT_DOCS}/{self.session_id}",
                                 data=json.dumps(doc_filenames))
        if response.status_code == 200:
            logger.info(response.json()["message"])
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def select_documents_by_keywords(self, keywords: List[str] | None = None):
        """Select a subset of documents in the collection the user has access to, based on the provided keywords.
        If the list of keywords is empty or None, the full collection of documents is selected for the RAG"""
        response = requests.post(url=f"{self.url}{ENDPOINT_SELECT_BY_KEYWORDS}/{self.session_id}",
                                 data=json.dumps(keywords))
        if response.status_code == 200:
            logger.info(response.json()["message"])
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def remove_documents(self, doc_filenames: List[str]) -> str:
        """Removes documents from the collection the user has access to, as well as associated embeddings"""
        response = requests.post(url=f"{self.url}{ENDPOINT_REMOVE_DOCS}/{self.session_id}",
                                 data=json.dumps(doc_filenames))
        if response.status_code == 200:
            logger.info(response.json())
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def list_available_docs(self) -> List[str]:
        """List available documents for a specific user"""
        response = requests.get(url=f"{self.url}{ENDPOINT_LIST_AVAILABLE_DOCS}/{self.session_id}")
        if response.status_code == 200:
            docs = json.loads(response.json())
            logger.info(f"Available docs: {docs}")
            return docs
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def query_rag(self, query: str) -> Dict:
        """Query the RAG and returns an answer"""
        # TODO: handle additional parameters to temporarily change the default config: number of retrieved docs & memory
        logger.debug(f"Question: {query}")
        stream = False
        if not stream:
            response = requests.get(url=self.url + ENDPOINT_QUERY_RAG,
                                    data=json.dumps({"query": query, "session_id": self.session_id}))
            if response.status_code == 200:
                rag_answer = json.loads(response.json())
                logger.debug(f"Response: {rag_answer}")
                return rag_answer
            else:
                logger.error(f"Error: {response.status_code, response.text}")
                raise RAGAPIError(f"Error: {response.status_code, response.text}")
        else:
            logger.debug("Response:")
            chunks = []
            with requests.get(url=self.url + ENDPOINT_QUERY_RAG,
                              data=json.dumps({"query": query, "session_id": self.session_id})) as r:
                for chunk in r.iter_lines():
                    chunks += chunk + "\n"
                    print(chunk)
            return chunks

    def get_current_sessions(self) -> List[str]:
        """Returns the identifiers of current sessions"""
        response = requests.get(url=self.url + ENDPOINT_CURRENT_SESSIONS)
        if response.status_code == 200:
            logger.debug(f"Current sessions: {response.json()}")
            return json.loads(response.json()).keys()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

def main():
    docs = [
        "NMT - Avenant n°4 de révision à l'accord sur le temps de travail.pdf", "NMT - Accord télétravail.pdf"
    ]

    files = [Path(__file__).parent.parent.parent.parent / "data" / "chatbot" / "drh" / doc for doc in docs]

    # create one client per user
    rag_client = RAGOrchestratorClient("alice", "drh")

    # upload data files
    rag_client.upload_files(files)
    rag_client.list_available_docs()

    # select document selection
    rag_client.select_documents_by_name([docs[1]])

    # query rag based on selected document collection
    rag_client.query_rag("Y'a une prime pour le télétravail?")

    # create one client per user
    rag_client2 = RAGOrchestratorClient("bob", "maintenance")
    rag_client2.upload_files([files[0]])
    rag_client2.query_rag("Y'a une prime pour le télétravail?")

    # delete doc
    rag_client2.remove_documents([docs[0]])

    # current sessions
    rag_client2.get_current_sessions()


if __name__ == "__main__":
    main()
