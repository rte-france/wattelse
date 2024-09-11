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

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CHECK_SERVICE,
    ENDPOINT_CREATE_SESSION,
    ENDPOINT_QUERY_RAG,
    ENDPOINT_UPLOAD_DOCS,
    ENDPOINT_REMOVE_DOCS,
    ENDPOINT_CURRENT_SESSIONS,
    ENDPOINT_CLEAR_COLLECTION,
    ENDPOINT_SELECT_BY_KEYWORDS,
    ENDPOINT_LIST_AVAILABLE_DOCS,
    ENDPOINT_CLEAN_SESSIONS,
    ENDPOINT_DOWNLOAD,
)


class RAGAPIError(Exception):
    pass


class RAGOrchestratorClient:
    """Class in charge of routing requests to right backend depending on user group"""

    def __init__(self, url: str = None):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "rag_orchestrator.cfg")
        self.port = config.get("RAG_ORCHESTRATOR_API_CONFIG", "port")
        self.host = config.get("RAG_ORCHESTRATOR_API_CONFIG", "host")
        self.url = f"http://{self.host}:{self.port}" if url is None else url
        if self.check_service():
            logger.debug("RAG Orchestrator is running")
        else:
            logger.error("Check RAG Orchestrator, does not seem to be running")

    def check_service(self) -> bool:
        """Check if RAG Orchestrator is running"""
        resp = requests.get(url=self.url + ENDPOINT_CHECK_SERVICE)
        return resp.json() == {"Status": "OK"}

    def create_session(self, group_id: str) -> str:
        """Create session associated to a group"""
        response = requests.post(self.url + ENDPOINT_CREATE_SESSION + f"/{group_id}")
        if response.status_code == 200:
            group_id = response.json()
            logger.info(f"[Group: {group_id}] RAGBackend created")
            return group_id
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def upload_files(self, group_id: str, file_paths: List[Path]):
        files = [("files", open(p, "rb")) for p in file_paths]

        response = requests.post(
            url=f"{self.url}{ENDPOINT_UPLOAD_DOCS}/{group_id}", files=files
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] {response.json()['message']}")
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def remove_documents(self, group_id: str, doc_filenames: List[str]) -> str:
        """Removes documents from the collection the user has access to, as well as associated embeddings"""
        response = requests.post(
            url=f"{self.url}{ENDPOINT_REMOVE_DOCS}/{group_id}",
            data=json.dumps(doc_filenames),
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] {response.json()['message']}")
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def clear_collection(self, group_id: str) -> str:
        """
        Clears all documents and embeddings from the collection.
        WARNING: all files will be lost permanently.
        """
        response = requests.post(
            url=f"{self.url}{ENDPOINT_CLEAR_COLLECTION}/{group_id}"
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] {response.json()['message']}")
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def list_available_docs(self, group_id: str) -> List[str]:
        """List available documents for a specific user"""
        response = requests.get(
            url=f"{self.url}{ENDPOINT_LIST_AVAILABLE_DOCS}/{group_id}"
        )
        if response.status_code == 200:
            docs = json.loads(response.json())
            logger.info(f"[Group: {group_id}] Available docs: {docs}")
            return docs
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def query_rag(
        self,
        group_id: str,
        message: str,
        history: List[Dict[str, str]] = None,
        secondary_system_prompt: str = None,
        selected_files: List[str] = None,
        stream: str = False,
    ) -> Dict:
        """Query the RAG and returns an answer"""
        # TODO: handle additional parameters to temporarily change the default config: number of retrieved docs & memory
        logger.debug(f"[Group: {group_id}] Question: {message}")

        response = requests.get(
            url=self.url + ENDPOINT_QUERY_RAG,
            data=json.dumps(
                {
                    "group_id": group_id,
                    "message": message,
                    "history": history,
                    "secondary_system_prompt": secondary_system_prompt,
                    "selected_files": selected_files,
                    "stream": stream,
                }
            ),
            stream=stream,
        )
        if response.status_code == 200:
            if stream:
                return response
            else:
                rag_answer = json.loads(response.json())
                logger.debug(f"[Group: {group_id}] Response: {rag_answer}")
                return rag_answer
        else:
            logger.error(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )
            raise RAGAPIError(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )

    def get_current_sessions(self) -> List[str]:
        """Returns current sessions ids"""
        response = requests.get(url=self.url + ENDPOINT_CURRENT_SESSIONS)
        if response.status_code == 200:
            logger.debug(f"Current sessions: {response.json()}")
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def clear_sessions(self, group_id: str | None = None):
        """
        Remove the specific session backend from RAG_SESSIONS.
        If no `session_id` is provided, remove all sessions' backend.
        """
        response = requests.post(url=f"{self.url}{ENDPOINT_CLEAN_SESSIONS}/{group_id}")
        if response.status_code == 200:
            logger.debug(
                f"[Group: {group_id}] Successfully removed {group_id if group_id else 'ALL'} session(s)"
            )
        else:
            logger.error(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )
            raise RAGAPIError(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )

    def download_to_dir(self, group_id: str, file_name: str, target_path: Path):
        """Downloads a collection file to the specified target path"""
        response = requests.get(
            url=f"{self.url}{ENDPOINT_DOWNLOAD}/{group_id}/{file_name}"
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] Successfully downloaded {file_name}")
            with open(target_path, "wb") as f:
                f.write(response.content)
        else:
            logger.error(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )
            raise RAGAPIError(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )
