import subprocess
from wattelse.api.rag_orchestrator.config.settings import CONFIG

# Start the FastAPI application
command = [
    "uvicorn",
    "wattelse.api.rag_orchestrator.main:app",
    "--host",
    CONFIG.host,
    "--port",
    str(CONFIG.port),
]
subprocess.run(command)
