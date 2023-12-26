from wattelse.common.vars import BASE_CACHE_PATH
from wattelse.common import BASE_DATA_DIR

DATA_DIR = BASE_DATA_DIR / "chatbot"
CACHE_DIR = BASE_CACHE_PATH / "chatbot"

# Make dirs if not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model parameters
MAX_TOKENS = 512
EMBEDDING_MODEL_NAME = "antoinelouis/biencoder-camembert-base-mmarcoFR"

# Retrieval modes
RETRIEVAL_DENSE = "dense"
RETRIEVAL_BM25 = "bm25"
RETRIEVAL_HYBRID = "hybrid"
RETRIEVAL_HYBRID_RERANKER = "hybrid+reranker"

# LLM API
LOCAL_LLM = "Local LLM"
CHATGPT_LLM = "ChatGPT"
