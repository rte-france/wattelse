import sys
import requests
from wattelse.api.embedding.config.settings import get_config

# Load config
CONFIG = get_config()

# Modify hostname if needed
if CONFIG.host == "0.0.0.0":
    CONFIG.host = "localhost"

# Check health endpoint
response = requests.get(f"http://{CONFIG.host}:{CONFIG.port}/health")

# Return status code
if response.status_code == 200:
    sys.exit(0)
else:
    sys.exit(1)
