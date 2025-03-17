import os

from wattelse.chatbot.frontend.config.settings import CONFIG

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.port})")
