import json
from fastapi import APIRouter, HTTPException

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CLEAR_COLLECTION,
    ENDPOINT_DOWNLOAD,
    ENDPOINT_LIST_AVAILABLE_DOCS,
    ENDPOINT_REMOVE_DOCS,
    RAG_SESSIONS,
)
from starlette.responses import FileResponse

from wattelse.api.rag_orchestrator.utils import require_session


router = APIRouter()


@router.post(ENDPOINT_REMOVE_DOCS + "/{group_id}")
@require_session
def remove_docs(group_id: str, doc_file_names: list[str]) -> dict[str, str]:
    """Remove the documents from raw storage and vector database"""
    RAG_SESSIONS[group_id].remove_docs(doc_file_names)
    return {
        "message": f"[Group: {group_id}] Successfully removed files {doc_file_names}"
    }


@router.post(ENDPOINT_CLEAR_COLLECTION + "/{group_id}")
@require_session
def clear_collection(group_id: str):
    """
    If collection exists, clears all documents and embeddings from the collection.
    WARNING: all files will be lost permanently.
    """
    RAG_SESSIONS[group_id].clear_collection()
    del RAG_SESSIONS[group_id]
    return {
        "message": f"[Group: {group_id}] Successfully cleared collection {group_id}"
    }


@router.get(ENDPOINT_LIST_AVAILABLE_DOCS + "/{group_id}")
@require_session
def list_available_docs(group_id: str) -> str:
    """List available documents for a specific user"""
    file_names = RAG_SESSIONS[group_id].get_available_docs()
    return json.dumps(file_names)


@router.get(ENDPOINT_DOWNLOAD + "/{group_id}/{file_name}")
@require_session
def download_file(group_id: str, file_name: str):
    if file_name in RAG_SESSIONS[group_id].get_available_docs():
        file_path = RAG_SESSIONS[group_id].get_file_path(file_name)
        if file_path:
            return FileResponse(
                file_path, media_type="application/octet-stream", filename=file_name
            )
    raise HTTPException(status_code=404, detail=f"File: {file_name} not found.")
