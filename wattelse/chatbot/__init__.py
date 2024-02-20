import configparser
import os

from pathlib import Path
from pydoc import locate

from wattelse.common.config_utils import parse_literal
from wattelse.common.vars import BASE_CACHE_PATH
from wattelse.common import BASE_DATA_DIR

# Ensures to write with +rw for both user and groups
os.umask(0o002)

DATA_DIR = BASE_DATA_DIR / "chatbot"
CACHE_DIR = BASE_CACHE_PATH / "chatbot"

# Make dirs if not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model parameters
MAX_TOKENS = 512

# Retrieval modes
RETRIEVAL_DENSE = "dense"
RETRIEVAL_BM25 = "bm25"
RETRIEVAL_HYBRID = "hybrid"
RETRIEVAL_HYBRID_RERANKER = "hybrid+reranker"

# LLM API
FASTCHAT_LLM = "Fastchat LLM"
OLLAMA_LLM = "Ollama LLM"
CHATGPT_LLM = "ChatGPT"

# Config for retriever and generator
config = configparser.ConfigParser(converters={"literal": parse_literal})
config.read(Path(__file__).parent / "config.cfg")

retriever_config =  parse_literal(dict(config["retriever"]))

generator_config = parse_literal(dict(config["generator"]))
TEMPERATURE = generator_config['temperature']
# resolve variable value
generator_config["custom_prompt"] = locate(generator_config["custom_prompt"])