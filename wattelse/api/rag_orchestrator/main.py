from fastapi import FastAPI
from wattelse.api.rag_orchestrator.routers import sessions, info, rag_query, documents

app = FastAPI()

app.include_router(info.router, tags=["Info"])
app.include_router(sessions.router, tags=["Sessions"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(rag_query.router, tags=["Queries"])
