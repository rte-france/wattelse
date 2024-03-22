import configparser
import os

from pathlib import Path
from pydoc import locate

from wattelse.common.config_utils import parse_literal

os.umask(0o002)

# Retriever methods
MMR = "mmr"
SIMILARITY = "similarity"
SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
BM25 = "bm25"
ENSEMBLE = "ensemble"

# LLM API
FASTCHAT_LLM = "Fastchat LLM"
OLLAMA_LLM = "Ollama LLM"
CHATGPT_LLM = "ChatGPT"
LLM_CONFIGS = {
    CHATGPT_LLM: Path(__file__).parent.parent.parent / "api" / "openai" / "openai.cfg",
    FASTCHAT_LLM: Path(__file__).parent.parent.parent / "api" / "fastchat" / "fastchat_api.cfg",
    OLLAMA_LLM: Path(__file__).parent.parent.parent / "api" / "ollama" / "ollama_api.cfg"
}

# Config for retriever and generator
config = configparser.ConfigParser(converters={"literal": parse_literal})
config.read(Path(__file__).parent / "rag_config.cfg")

retriever_config =  parse_literal(dict(config["retriever"]))

generator_config = parse_literal(dict(config["generator"]))
# resolve variable value
generator_config["custom_prompt"] = locate(generator_config["custom_prompt"])
