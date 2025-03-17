from fastapi import APIRouter

from wattelse.api.rag_orchestrator import (
    ENDPOINT_HEALTH_SERVICE,
    ENDPOINT_GENERATION_MODEL_NAME,
)

from wattelse.api.rag_orchestrator.rag_sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session


router = APIRouter()


@router.get(ENDPOINT_HEALTH_SERVICE)
def home():
    """Returns the status of the service."""
    return {"status": "ok"}


@router.get(ENDPOINT_GENERATION_MODEL_NAME + "/{group_id}")
@require_session
def get_llm_name(group_id: str):
    return RAG_SESSIONS[group_id].get_llm_model_name()
