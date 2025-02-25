import json
from fastapi import APIRouter
from loguru import logger

from wattelse.api.rag_orchestrator import ENDPOINT_QUERY_RAG
from wattelse.api.rag_orchestrator.models import RAGQuery
from starlette.responses import StreamingResponse

from wattelse.api.rag_orchestrator.routers.sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session, data_streamer


router = APIRouter()


@router.post(ENDPOINT_QUERY_RAG + "/{group_id}")
@require_session
async def query_rag(group_id: str, rag_query: RAGQuery) -> StreamingResponse:
    """Query the RAG and returns the answer and associated sources"""
    logger.debug(f"[Group: {group_id}] Received query: {rag_query.message}")
    response = RAG_SESSIONS[group_id].query_rag(
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
