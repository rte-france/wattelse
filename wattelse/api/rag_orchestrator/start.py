import uvicorn
from wattelse.api.rag_orchestrator.config.settings import CONFIG

# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "wattelse.api.rag_orchestrator.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=True,
    )
