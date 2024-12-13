#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json

from loguru import logger
from typing import List, Dict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CHECK_SERVICE,
    ENDPOINT_CREATE_SESSION,
    ENDPOINT_QUERY_RAG,
    ENDPOINT_UPLOAD_DOCS,
    ENDPOINT_REMOVE_DOCS,
    ENDPOINT_CURRENT_SESSIONS,
    ENDPOINT_LIST_AVAILABLE_DOCS,
    ENDPOINT_CLEAN_SESSIONS,
    ENDPOINT_DOWNLOAD,
    ENDPOINT_CLEAR_COLLECTION,
    ENDPOINT_GENERATION_MODEL_NAME,
)
from wattelse.chatbot.backend.rag_backend import RAGBackEnd

SESSION_TIMEOUT = 30  # in minutes

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
    group_system_prompt: str | None
    selected_files: List[str] | None
    stream: bool = False


@app.get(ENDPOINT_CHECK_SERVICE)
def home():
    """Returns the status of the service."""
    return {"Status": "OK"}


@app.post(ENDPOINT_CREATE_SESSION + "/{group_id}")
def create_session(group_id: str, config_file_path: str) -> str:
    """When this is called, instantiates a RAG backend for a group."""
    if group_id not in RAG_SESSIONS.keys():
        RAG_SESSIONS[group_id] = RAGBackEnd(group_id, Path(config_file_path))
        logger.info(f"[Group: {group_id}] RAGBackend created")
    else:
        logger.warning(f"[Group: {group_id}] RAGBackend already created")
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
        logger.debug(f"[Group: {group_id}] Uploading file: {file.filename}...")
        # Check if the file is already in the document collection
        if collection.is_present(file.filename):
            logger.warning(
                f"[Group: {group_id}] File {file.filename} already present in the collection {collection_name}, "
                f"skipping indexing and chunking"
            )
            continue

        RAG_SESSIONS[group_id].add_file_to_collection(file.filename, file.file)

    return {
        "message": f"[Group: {group_id}] Successfully uploaded {[file.filename for file in files]}"
    }


@app.post(ENDPOINT_REMOVE_DOCS + "/{group_id}")
def remove_docs(group_id: str, doc_file_names: List[str]) -> Dict[str, str]:
    """Remove the documents from raw storage and vector database"""
    check_if_session_exists(group_id)
    RAG_SESSIONS[group_id].remove_docs(doc_file_names)
    return {
        "message": f"[Group: {group_id}] Successfully removed files {doc_file_names}"
    }


@app.post(ENDPOINT_CLEAR_COLLECTION + "/{group_id}")
def clear_collection(group_id: str):
    """
    If collection exists, clears all documents and embeddings from the collection.
    WARNING: all files will be lost permanently.
    """
    if group_id in RAG_SESSIONS.keys():
        RAG_SESSIONS[group_id].clear_collection()
        del RAG_SESSIONS[group_id]
        return {
            "message": f"[Group: {group_id}] Successfully cleared collection {group_id}"
        }
    else:
        return {"message": f"[Group: {group_id}] Session not found"}


@app.get(ENDPOINT_LIST_AVAILABLE_DOCS + "/{group_id}")
def list_available_docs(group_id: str) -> str:
    """List available documents for a specific user"""
    check_if_session_exists(group_id)
    file_names = RAG_SESSIONS[group_id].get_available_docs()
    return json.dumps(file_names)


def data_streamer(stream_data):
    """Generator to stream response from RAGBackend to RAG client.
    Encodes received chunks in a binary format and streams them.
    """
    for i in stream_data:
        yield f"{i}".encode("utf-8")


@app.post(ENDPOINT_QUERY_RAG)
async def query_rag(rag_query: RAGQuery) -> StreamingResponse:
    """Query the RAG and returns the answer and associated sources"""
    check_if_session_exists(rag_query.group_id)
    logger.debug(f"[Group: {rag_query.group_id}] Received query: {rag_query.message}")
    response = RAG_SESSIONS[rag_query.group_id].query_rag(
        rag_query.message,
        history=rag_query.history,
        group_system_prompt=rag_query.group_system_prompt,
        selected_files=rag_query.selected_files,
        stream=rag_query.stream,
    )
    if rag_query.stream:
        return StreamingResponse(
            data_streamer(response), media_type="text/event-stream"
        )
    else:
        return json.dumps(response)


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
            raise HTTPException(
                status_code=404, detail=f"Group id : {group_id} not found in sessions."
            )
    else:
        RAG_SESSIONS.clear()


def check_if_session_exists(group_id: str):
    """Check if the session_id exists, if not throw an exception"""
    if group_id not in RAG_SESSIONS:
        raise RAGOrchestratorAPIError(
            f"Session {group_id} does not exist, please reconnect"
        )


@app.get(ENDPOINT_DOWNLOAD + "/{group_id}/{file_name}")
def download_file(group_id: str, file_name: str):
    check_if_session_exists(group_id)
    if file_name in RAG_SESSIONS[group_id].get_available_docs():
        file_path = RAG_SESSIONS[group_id].get_file_path(file_name)
        if file_path:
            return FileResponse(
                file_path, media_type="application/octet-stream", filename=file_name
            )
    raise HTTPException(status_code=404, detail=f"File: {file_name} not found.")


@app.get(ENDPOINT_GENERATION_MODEL_NAME + "/{group_id}")
def get_llm_name(group_id: str):
    check_if_session_exists(group_id)
    return RAG_SESSIONS[group_id].get_llm_model_name()


# to run the API (reload each time the python is changed)
# uvicorn rag_orchestrator_api:app --reload
