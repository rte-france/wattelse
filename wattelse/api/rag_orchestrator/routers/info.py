from fastapi import APIRouter

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CHECK_SERVICE,
    ENDPOINT_GENERATION_MODEL_NAME,
    RAG_SESSIONS,
)
from wattelse.api.rag_orchestrator.utils import require_session


router = APIRouter()


@router.get(ENDPOINT_CHECK_SERVICE)
def home():
    """Returns the status of the service."""
    return {"Status": "OK"}


@router.get(ENDPOINT_GENERATION_MODEL_NAME + "/{group_id}")
@require_session
def get_llm_name(group_id: str):
    return RAG_SESSIONS[group_id].get_llm_model_name()
