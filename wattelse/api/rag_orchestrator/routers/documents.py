#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
from loguru import logger
from fastapi import APIRouter, File, HTTPException, UploadFile, Security
from starlette import status

from wattelse.api.rag_orchestrator import (
    ENDPOINT_CLEAR_COLLECTION,
    ENDPOINT_DOWNLOAD,
    ENDPOINT_LIST_AVAILABLE_DOCS,
    ENDPOINT_REMOVE_DOCS,
    ENDPOINT_UPLOAD_DOCS,
)
from starlette.responses import FileResponse


from wattelse.api.rag_orchestrator.rag_sessions import RAG_SESSIONS
from wattelse.api.rag_orchestrator.utils import require_session
from wattelse.api.security import (
    TokenData,
    get_current_client,
    RESTRICTED_ACCESS,
    is_authorized_for_group,
)

router = APIRouter()


@router.post(ENDPOINT_UPLOAD_DOCS + "/{group_id}")
@require_session
def upload(
    group_id: str,
    files: list[UploadFile] = File(...),
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
):
    """Upload a list of documents into a document collection"""
    logger.debug(f"Request to {ENDPOINT_UPLOAD_DOCS} from: {current_client.client_id}")
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )
    # Get current document collection
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


@router.post(ENDPOINT_REMOVE_DOCS + "/{group_id}")
@require_session
def remove_docs(
    group_id: str,
    doc_file_names: list[str],
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
) -> dict[str, str]:
    """Remove the documents from raw storage and vector database"""
    logger.debug(f"Request to {ENDPOINT_REMOVE_DOCS} from: {current_client.client_id}")
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )
    RAG_SESSIONS[group_id].remove_docs(doc_file_names)
    return {
        "message": f"[Group: {group_id}] Successfully removed files {doc_file_names}"
    }


@router.post(ENDPOINT_CLEAR_COLLECTION + "/{group_id}")
@require_session
def clear_collection(
    group_id: str,
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
):
    """
    If collection exists, clears all documents and embeddings from the collection.
    WARNING: all files will be lost permanently.
    """
    logger.debug(
        f"Request to {ENDPOINT_CLEAR_COLLECTION} from: {current_client.client_id}"
    )
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )

    RAG_SESSIONS[group_id].clear_collection()
    del RAG_SESSIONS[group_id]
    return {
        "message": f"[Group: {group_id}] Successfully cleared collection {group_id}"
    }


@router.get(
    ENDPOINT_LIST_AVAILABLE_DOCS + "/{group_id}",
)
@require_session
def list_available_docs(
    group_id: str,
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
) -> str:
    """List available documents for a specific user"""
    logger.debug(
        f"Request to {ENDPOINT_LIST_AVAILABLE_DOCS} from: {current_client.client_id}"
    )
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )

    file_names = RAG_SESSIONS[group_id].get_available_docs()
    return json.dumps(file_names)


@router.get(
    ENDPOINT_DOWNLOAD + "/{group_id}/{file_name}",
)
@require_session
def download_file(
    group_id: str,
    file_name: str,
    current_client: TokenData = Security(
        get_current_client, scopes=[RESTRICTED_ACCESS]
    ),
):
    logger.debug(f"Request to {ENDPOINT_DOWNLOAD} from: {current_client.client_id}")
    if not is_authorized_for_group(current_client, group_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client does not authorized for group {group_id}",
        )

    if file_name in RAG_SESSIONS[group_id].get_available_docs():
        file_path = RAG_SESSIONS[group_id].get_file_path(file_name)
        if file_path:
            return FileResponse(
                file_path, media_type="application/octet-stream", filename=file_name
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"File: {file_name} not found."
    )
