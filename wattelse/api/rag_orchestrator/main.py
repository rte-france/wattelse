from fastapi import FastAPI
from wattelse.api.rag_orchestrator.routers import info, rag_query, sessions, documents

app = FastAPI()

app.include_router(info.router, tags=["Utils"])
app.include_router(sessions.router, tags=["Sessions"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(rag_query.router, tags=["Queries"])
