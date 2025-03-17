import os

from wattelse.api.vllm.config.settings import get_config

# Load configuration
CONFIG = get_config()

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.port})")
