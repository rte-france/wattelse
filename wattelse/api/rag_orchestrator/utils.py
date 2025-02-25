from functools import wraps
import inspect
from fastapi import HTTPException
from wattelse.api.rag_orchestrator.routers.sessions import RAG_SESSIONS


def require_session(func):
    """
    Decorator to check if a session exists before calling the endpoint.
    Handle both sync and async functions.
    """
    # Check if the function is async or not
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(group_id: str, *args, **kwargs):
            if group_id not in RAG_SESSIONS.keys():
                raise HTTPException(
                    status_code=404,
                    detail=f"Session with group ID `{group_id}` not found. Please create a session first.",
                )
            return await func(group_id, *args, **kwargs)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(group_id: str, *args, **kwargs):
            if group_id not in RAG_SESSIONS.keys():
                raise HTTPException(
                    status_code=404,
                    detail=f"Session with group ID `{group_id}` not found. Please create a session first.",
                )
            return func(group_id, *args, **kwargs)

        return sync_wrapper


def data_streamer(stream_data):
    """Generator to stream response from RAGBackend to RAG client.
    Encodes received chunks in a binary format and streams them.
    """
    for i in stream_data:
        yield f"{i}".encode("utf-8")
