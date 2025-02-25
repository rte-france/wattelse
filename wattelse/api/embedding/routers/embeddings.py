from fastapi import APIRouter
from loguru import logger
from sentence_transformers import SentenceTransformer

from wattelse.api.embedding.config.config import CONFIG
from wattelse.api.embedding.models import InputText

# Load embedding model
logger.debug(f"Loading embedding model : {CONFIG.model_name}")
EMBEDDING_MODEL = SentenceTransformer(CONFIG.model_name)

# Fix max model length error
if EMBEDDING_MODEL.max_seq_length == 514:
    EMBEDDING_MODEL.max_seq_length = 512
router = APIRouter()


@router.post("/encode")
def embed(input: InputText):
    emb = EMBEDDING_MODEL.encode(input.text, show_progress_bar=input.show_progress_bar)
    return {"embeddings": emb.tolist()}
