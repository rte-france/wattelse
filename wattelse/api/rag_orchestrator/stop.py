import os

from wattelse.api.rag_orchestrator.config.settings import CONFIG

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.port})")
