#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
from fastapi import APIRouter, Security, HTTPException
from loguru import logger
from starlette import status

from wattelse.api.rag_orchestrator import ENDPOINT_QUERY_RAG
from wattelse.api.rag_orchestrator.models import RAGQuery
from starlette.responses import StreamingResponse


from wattelse.api.rag_orchestrator.rag_sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session, data_streamer
from wattelse.api.common.security import (
    get_current_client,
    TokenData,
    RESTRICTED_ACCESS,
    is_authorized_for_group,
)

router = APIRouter()


@router.post(
    ENDPOINT_QUERY_RAG + "/{group_id}",
    summary="Query the RAG and returns the answer and associated sources (requires restricted_access scope)",
)
@require_session
async def query_rag(
    group_id: str,
    rag_query: RAGQuery,
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
) -> StreamingResponse:
    """Query the RAG and returns the answer and associated sources"""
    logger.debug(f"Request to {ENDPOINT_QUERY_RAG} from: {current_client.client_id}")
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )

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
