import os

from wattelse.api.embedding.config.settings import get_config

# Load the configuration
CONFIG = get_config()

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.port})")
