#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import os

import requests
from pathlib import Path

from loguru import logger

from wattelse.api.common.client import APIClient
from wattelse.api.rag_orchestrator import (
    ENDPOINT_HEALTH_SERVICE,
    ENDPOINT_CREATE_SESSION,
    ENDPOINT_QUERY_RAG,
    ENDPOINT_UPLOAD_DOCS,
    ENDPOINT_REMOVE_DOCS,
    ENDPOINT_CURRENT_SESSIONS,
    ENDPOINT_CLEAR_COLLECTION,
    ENDPOINT_LIST_AVAILABLE_DOCS,
    ENDPOINT_CLEAN_SESSIONS,
    ENDPOINT_DOWNLOAD,
    ENDPOINT_GENERATION_MODEL_NAME,
)
from wattelse.api.rag_orchestrator.models import RAGConfig


class RAGAPIError(Exception):
    pass


class RAGOrchestratorClient(APIClient):
    """Class in charge of routing requests to right backend depending on user group"""

    def __init__(
        self,
        url: str | None = "https://localhost:1978",
        client_id: str = "wattelse",
        client_secret: str = os.getenv("WATTELSE_CLIENT_SECRET", None),
    ):
        super().__init__(url, client_id, client_secret)
        try:
            if self.check_service():
                logger.debug("RAG Orchestrator is running")
            else:
                logger.warning("RAGOrchestrator API is not running")
        except:
            logger.warning("RAGOrchestrator API is not running")

    def check_service(self) -> bool:
        """Check if RAG Orchestrator is running"""
        resp = requests.get(url=self.url + ENDPOINT_HEALTH_SERVICE, verify=False)
        return resp.json() == {"status": "ok"}

    def get_rag_llm_model(self, group_id: str) -> str:
        """Returns the name of the LLM used by the RAG"""
        response = requests.get(
            url=f"{self.url}{ENDPOINT_GENERATION_MODEL_NAME}/{group_id}", verify=False
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] LLM model name for RAG {response}")
            return response.json()
        else:
            logger.error(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )
            raise RAGAPIError(
                f"[Group: {group_id}] Error: {response.status_code, response.text}"
            )

    def create_session(self, group_id: str, config: RAGConfig) -> str:
        """Create session associated to a group"""
        response = requests.post(
            self.url + ENDPOINT_CREATE_SESSION + f"/{group_id}",
            json={"config": config},
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code == 200:
            logger.info(response.json()["message"])
            return group_id
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            raise RAGAPIError(f"Error {response.status_code}: {response.text}")

    def upload_files(self, group_id: str, file_paths: list[Path]):
        files = [("files", open(p, "rb")) for p in file_paths]

        response = requests.post(
            url=f"{self.url}{ENDPOINT_UPLOAD_DOCS}/{group_id}",
            files=files,
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] {response.json()['message']}")
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def remove_documents(self, group_id: str, doc_filenames: list[str]) -> str:
        """Removes documents from the collection the user has access to, as well as associated embeddings"""
        response = requests.post(
            url=f"{self.url}{ENDPOINT_REMOVE_DOCS}/{group_id}",
            data=json.dumps(doc_filenames),
            verify=False,
            headers=self._get_headers(),
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
            url=f"{self.url}{ENDPOINT_CLEAR_COLLECTION}/{group_id}",
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code == 200:
            logger.debug(f"[Group: {group_id}] {response.json()['message']}")
            return response.json()
        else:
            logger.error(f"Error: {response.status_code, response.text}")
            raise RAGAPIError(f"Error: {response.status_code, response.text}")

    def list_available_docs(self, group_id: str) -> list[str]:
        """List available documents for a specific user"""
        response = requests.get(
            url=f"{self.url}{ENDPOINT_LIST_AVAILABLE_DOCS}/{group_id}",
            verify=False,
            headers=self._get_headers(),
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
        history: list[dict[str, str]] = None,
        group_system_prompt: str = None,
        selected_files: list[str] = None,
        stream: str = False,
    ) -> dict:
        """Query the RAG and returns an answer"""
        # TODO: handle additional parameters to temporarily change the default config: number of retrieved docs & memory
        logger.debug(f"[Group: {group_id}] Question: {message}")

        response = requests.post(
            url=self.url + ENDPOINT_QUERY_RAG + "/" + group_id,
            data=json.dumps(
                {
                    "message": message,
                    "history": history,
                    "group_system_prompt": group_system_prompt,
                    "selected_files": selected_files,
                    "stream": stream,
                }
            ),
            stream=stream,
            verify=False,
            headers=self._get_headers(),
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

    def get_current_sessions(self) -> list[str]:
        """Returns current sessions ids"""
        response = requests.get(url=self.url + ENDPOINT_CURRENT_SESSIONS, verify=False)
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
        response = requests.post(
            url=f"{self.url}{ENDPOINT_CLEAN_SESSIONS}/{group_id}",
            verify=False,
            headers=self._get_headers(),
        )
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
            url=f"{self.url}{ENDPOINT_DOWNLOAD}/{group_id}/{file_name}",
            verify=False,
            headers=self._get_headers(),
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
