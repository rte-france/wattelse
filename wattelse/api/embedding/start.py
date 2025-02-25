import os
import subprocess
from wattelse.api.embedding.config.settings import CONFIG

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.cuda_visible_devices

# Start the FastAPI application
command = [
    "uvicorn",
    "wattelse.api.embedding.main:app",
    "--host",
    CONFIG.host,
    "--port",
    str(CONFIG.port),
    "--workers",
    str(CONFIG.number_workers),
]
subprocess.run(command)
