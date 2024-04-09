#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from wattelse.api.rag_orchestrator import ENDPOINT_CHECK_SERVICE, ENDPOINT_CREATE_SESSION, \
    ENDPOINT_QUERY_RAG, ENDPOINT_UPLOAD_DOCS, ENDPOINT_REMOVE_DOCS, ENDPOINT_CURRENT_SESSIONS, \
    ENDPOINT_SELECT_BY_KEYWORDS, ENDPOINT_LIST_AVAILABLE_DOCS, ENDPOINT_CLEAN_SESSIONS
from wattelse.chatbot.backend.rag_backend import RAGBackEnd

SESSION_TIMEOUT = 30  # in minutes

API_CONFIG_FILE_PATH = Path(__file__).parent / "rag_orchestrator_api.cfg"
config = configparser.ConfigParser()
config.read(API_CONFIG_FILE_PATH)

# Initialize API
app = FastAPI()

# management of sessions
RAG_SESSIONS: Dict[str, RAGBackEnd] = {}  # used to link a backend to a group

class RAGOrchestratorAPIError(Exception):
    """Generic exception for RAG orchestrator API"""
    pass


class RAGQuery(BaseModel):
    group_id: str
    message: str
    history: List[Dict[str, str]] | None
    selected_files: List[str] | None
    


@app.get(ENDPOINT_CHECK_SERVICE)
def home():
    """Returns the status of the service."""
    return {"Status": "OK"}


@app.post(ENDPOINT_CREATE_SESSION + "/{group_id}")
def create_session(group_id: str) -> str:
    """When this is called, instantiates a RAG backend for a group."""
    if group_id not in RAG_SESSIONS.keys():
        RAG_SESSIONS[group_id] = RAGBackEnd(group_id)
    logger.info(f"Created a RAGBackend for group {group_id}")
    return group_id


@app.get(ENDPOINT_CURRENT_SESSIONS)
def current_sessions() -> List[str]:
    """Returns the list of current active sessions"""
    return list(RAG_SESSIONS.keys())


@app.post(ENDPOINT_UPLOAD_DOCS + "/{group_id}")
def upload(group_id: str, files: List[UploadFile] = File(...)):
    """Upload a list of documents into a document collection"""
    # get current document collection
    check_if_session_exists(group_id)
    collection = RAG_SESSIONS[group_id].document_collection
    collection_name = collection.collection_name

    for file in files:
        logger.debug(f"[group_id: {group_id}] Uploading file: {file.filename}...")
        # Check if the file is already in the document collection
        if collection.is_present(file.filename):
            logger.warning(
                f"File {file.filename} already present in the collection {collection_name}, skippping indexing and chunking")
            continue

        RAG_SESSIONS[group_id].add_file_to_collection(file)

    return {"message": f"[group_id: {group_id}] Successfully uploaded {[file.filename for file in files]}"}


# @app.post(ENDPOINT_SELECT_BY_KEYWORDS + "/{session_id}")
# def select_by_keywords(session_id: str, keywords: List[str] | None):
#     """Select the documents to be used for the RAG among those the user have access to; if nothing is provided,
#     uses all acessible documents"""
#     logger.debug(f"List of selected keywords: {keywords}")
#     check_if_session_exists(session_id)
#     RAG_sessions[session_id].select_by_keywords(keywords)
#     update_session_usage(session_id)
#     return {
#         "message": f"[session_id: {session_id}] Successfully filtered document collection based on keywords {keywords}"}


@app.post(ENDPOINT_REMOVE_DOCS + "/{group_id}")
def remove_docs(group_id: str, doc_file_names: List[str]) -> Dict[str, str]:
    """Remove the documents from raw storage and vector database"""
    check_if_session_exists(group_id)
    RAG_SESSIONS[group_id].remove_docs(doc_file_names)
    return {"message": f"[group_id: {group_id}] Successfully removed files {doc_file_names}"}


@app.get(ENDPOINT_LIST_AVAILABLE_DOCS + "/{group_id}")
def list_available_docs(group_id: str) -> str:
    """List available documents for a specific user"""
    check_if_session_exists(group_id)
    file_names = RAG_SESSIONS[group_id].get_available_docs()
    return json.dumps(file_names)


@app.get(ENDPOINT_QUERY_RAG)
def query_rag(rag_query: RAGQuery) -> str:
    """Query the RAG and returns the answer and associated sources"""
    logger.debug(f"Received query: {rag_query.message}")
    check_if_session_exists(rag_query.group_id)
    # TODO: stream response, cf https://www.vidavolta.io/streaming-with-fastapi/
    return json.dumps(
        RAG_SESSIONS[rag_query.group_id].query_rag(
            rag_query.message,
            history=rag_query.history,
            selected_files=rag_query.selected_files,
            )
        )

@app.post(ENDPOINT_CLEAN_SESSIONS + "/{group_id}")
def clean_sessions(group_id: str | None = None):
    """
    Remove the specific `group_id` backend from RAG_SESSIONS.
    If no `group_id` is provided, remove all sessions backend.
    """
    if group_id:
        if group_id in RAG_SESSIONS.keys():
            del RAG_SESSIONS[group_id]
        else:
            raise HTTPException(status_code=404, detail=f"Group id : {group_id} not found in sessions.")
    else:
        RAG_SESSIONS.clear()


def check_if_session_exists(group_id: str):
    """Check if the session_id exists, if not throw an exception"""
    if group_id not in RAG_SESSIONS:
        raise RAGOrchestratorAPIError(f"Session {group_id} does not exist, please reconnect")


# to run the API (reload each time the python is changed)
# uvicorn rag_orchestrator_api:app --reload