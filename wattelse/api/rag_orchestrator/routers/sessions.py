#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from fastapi import APIRouter, HTTPException, Security
from loguru import logger
from starlette import status

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CLEAN_SESSIONS,
    ENDPOINT_CREATE_SESSION,
    ENDPOINT_CURRENT_SESSIONS,
)
from wattelse.api.rag_orchestrator.models import RAGConfig
from wattelse.api.rag_orchestrator.rag_sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session
from wattelse.api.security import (
    RESTRICTED_ACCESS,
    FULL_ACCESS,
    get_current_client,
    TokenData,
    is_authorized_for_group,
)
from wattelse.chatbot.backend.rag_backend import RAGBackend


router = APIRouter()


@router.get(ENDPOINT_CURRENT_SESSIONS)
def current_sessions() -> list[str]:
    """Returns the list of current active sessions"""
    return list(RAG_SESSIONS.keys())


@router.post(
    ENDPOINT_CREATE_SESSION + "/{group_id}",
    summary="Instantiate a RAG backend for the specified group (requires restricted_access scope)",
)
def create_session(
    group_id: str,
    config: RAGConfig,
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
) -> dict:
    """When this is called, instantiates a RAG backend for a group."""
    logger.debug(
        f"Request to {ENDPOINT_CREATE_SESSION} from: {current_client.client_id}"
    )
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )

    if group_id not in RAG_SESSIONS.keys():
        try:
            RAG_SESSIONS[group_id] = RAGBackend(
                group_id,
                config=config.config,
            )
            logger.info(f"[Group: {group_id}] RAGBackend created")
            return {"message": f"Session for group '{group_id}' created"}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
    else:
        logger.warning(f"[Group: {group_id}] RAGBackend already created")
        return {"message": f"Session for group '{group_id}' already exists"}


@router.post(
    ENDPOINT_CLEAN_SESSIONS + "/{group_id}",
    summary="Remove the specific `group_id` backend from RAG_SESSIONS. If no `group_id` is provided, remove all sessions backend (requires full_access scope)",
)
@require_session
def clean_sessions(
    group_id: str | None = None,
    current_client: TokenData = Security(get_current_client, scopes=[FULL_ACCESS]),
):
    """
    Remove the specific `group_id` backend from RAG_SESSIONS.
    If no `group_id` is provided, remove all sessions backend.
    """
    logger.debug(
        f"Request to {ENDPOINT_CLEAN_SESSIONS} from: {current_client.client_id}"
    )
    if group_id:
        if group_id in RAG_SESSIONS.keys():
            del RAG_SESSIONS[group_id]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group id : {group_id} not found in sessions.",
            )
    else:
        RAG_SESSIONS.clear()
