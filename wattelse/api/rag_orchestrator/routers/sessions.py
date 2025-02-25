from fastapi import APIRouter, HTTPException
from loguru import logger

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CLEAN_SESSIONS,
    ENDPOINT_CREATE_SESSION,
    ENDPOINT_CURRENT_SESSIONS,
)
from wattelse.api.rag_orchestrator.models import RAGConfig
from wattelse.api.rag_orchestrator.rag_sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session
from wattelse.chatbot.backend.rag_backend import RAGBackend


router = APIRouter()


@router.get(ENDPOINT_CURRENT_SESSIONS)
def current_sessions() -> list[str]:
    """Returns the list of current active sessions"""
    return list(RAG_SESSIONS.keys())


@router.post(ENDPOINT_CREATE_SESSION + "/{group_id}")
def create_session(group_id: str, config: RAGConfig) -> dict:
    """When this is called, instantiates a RAG backend for a group."""
    if group_id not in RAG_SESSIONS.keys():
        try:
            RAG_SESSIONS[group_id] = RAGBackend(
                group_id,
                config=config.config,
            )
            logger.info(f"[Group: {group_id}] RAGBackend created")
            return {"message": f"Session for group '{group_id}' created"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.warning(f"[Group: {group_id}] RAGBackend already created")
        return {"message": f"Session for group '{group_id}' already exists"}


@router.post(ENDPOINT_CLEAN_SESSIONS + "/{group_id}")
@require_session
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
