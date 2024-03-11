import configparser
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from timeloop import Timeloop

from wattelse.api.rag_orchestrator import ENDPOINT_CHECK_SERVICE, ENDPOINT_CREATE_SESSION, ENDPOINT_SELECT_DOCS, \
    ENDPOINT_QUERY_RAG, ENDPOINT_UPLOAD_DOCS, ENDPOINT_REMOVE_DOCS, ENDPOINT_CURRENT_SESSIONS, ENDPOINT_CHAT_HISTORY
from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.chatbot import (
    retriever_config,
    generator_config, )

SESSION_TIMEOUT = 15  # in minutes

API_CONFIG_FILE_PATH = Path(__file__).parent / "rag_orchestrator_api.cfg"
config = configparser.ConfigParser()
config.read(API_CONFIG_FILE_PATH)

# initialize timeloop
tl = Timeloop()

# initialize API
app = FastAPI()

# management of sessions
RAG_sessions: Dict[str, RAGBackEnd] = {}  # used to link a backend to a session ID
RAG_sessions_usage: Dict[str, Dict[str, datetime]] = {}  # used to clean sessions


# Pydantic class for FastAPI typing control
class User(BaseModel):
    login: str


class RAGQuery(BaseModel):
    query: str
    session_id: str


@app.get(ENDPOINT_CHECK_SERVICE)
def home():
    """Returns the status of the service."""
    return {"Status": "OK"}


@app.post(ENDPOINT_CREATE_SESSION)
def create_session(user: User) -> str:
    """When this is called, instantiates a RAG backend, and associate a session id to this backend"""
    # TODO: parameters to customize the backend can be added
    now = datetime.now()
    session_id = str(uuid.uuid1())
    RAG_sessions[session_id] = RAGBackEnd(user.login, **retriever_config, **generator_config)
    RAG_sessions_usage[session_id] = {"created": now, "last_used": now}
    logger.info(f"Session id: {session_id} for user {user.login}")
    return session_id


@app.get(ENDPOINT_CURRENT_SESSIONS)
def current_sessions() -> str:
    """Returns the list of current active sessions"""
    return json.dumps(RAG_sessions_usage, default=str)


@app.post(ENDPOINT_UPLOAD_DOCS + "/{session_id}")
def upload(session_id: str, files: List[UploadFile] = File(...)):
    """Upload a list of documents into a document collection"""
    # get current document collection
    collection = RAG_sessions[session_id].document_collection
    collection_name = collection.collection_name

    for file in files:
        logger.debug(f"[session_id: {session_id}] Uploading file: {file.filename}...")
        # Check if the file is already in the document collection
        if collection.is_present(file.filename):
            logger.warning(
                f"File {file.filename} already present in the collection {collection_name}, skippping indexing and chunking")
            continue

        RAG_sessions[session_id].add_file_to_collection(file)

        # Update session
        update_session_usage(session_id)

    return {"message": f"[session_id: {session_id}] Successfully uploaded {[file.filename for file in files]}"}


@app.post(ENDPOINT_SELECT_DOCS + "/{session_id}")
def select_docs(session_id: str, doc_file_names: List[str] | None):
    """Select the documents to be used for the RAG among those the user have access to; if nothing is provided,
    uses all acessible documents"""
    logger.debug(f"List of selected docs: {doc_file_names}")
    RAG_sessions[session_id].select_docs(doc_file_names)
    update_session_usage(session_id)
    return {"message": f"[session_id: {session_id}] Successfully selected files {doc_file_names}"}


@app.post(ENDPOINT_REMOVE_DOCS + "/{session_id}")
def remove_docs(session_id: str, doc_file_names: List[str]) -> Dict[str, str]:
    """Remove the documents from raw storage and vector database"""
    RAG_sessions[session_id].remove_docs(doc_file_names)
    update_session_usage(session_id)
    return {"message": f"[session_id: {session_id}] Successfully removed files {doc_file_names}"}


@app.get(ENDPOINT_QUERY_RAG)
def query_rag(rag_query: RAGQuery) -> str:
    """Query the RAG and returns the answer and associated sources"""
    logger.debug(f"Received query: {rag_query.query}")
    update_session_usage(rag_query.session_id)
    # TODO: stream response, cf https://www.vidavolta.io/streaming-with-fastapi/
    return RAG_sessions[rag_query.session_id].query_rag(rag_query.query)


@app.get(ENDPOINT_CHAT_HISTORY + "/{login}")
def get_chat_history(login: str) -> str:
    """Returns the chat history associated to a user"""
    #TODO: à implémenter
    logger.debug(f"Received")
    return "Not implemented yet"


def update_session_usage(session_id: str):
    """Update the session usage, used for cleanup"""
    RAG_sessions_usage[session_id]["last_used"] = datetime.now()


@tl.job(interval=timedelta(minutes=60))
def clean_sessions():
    """Clean user sessions periodically in order to consume too much memory"""
    # TODO: not implemented yet use info contained in RAG_sessions_usage
    logger.info("Cleaning sessions...")

# to run the API (reload each time the python is changed)
# uvicorn rag_orchestrator_api:app --reload
