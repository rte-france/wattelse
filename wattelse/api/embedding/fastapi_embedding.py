import configparser
from pathlib import Path
from loguru import logger
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

EMBEDDING_API_CONFIG_FILE_PATH = Path(__file__).parent / "embedding_api.cfg"
config = configparser.ConfigParser()
config.read(EMBEDDING_API_CONFIG_FILE_PATH)
EMBEDDING_MODEL_NAME = config.get('EMBEDDING_API_CONFIG', 'model_name')

logger.debug(f'Loading embedding model : {EMBEDDING_MODEL_NAME}')
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Fix max model length error
if EMBEDDING_MODEL.max_seq_length == 514:
        EMBEDDING_MODEL.max_seq_length = 512

# Pydantic class for FastAPI typing control
class InputText(BaseModel):
    text: str | List[str]

app = FastAPI()

@app.post('/encode')
def embed(input: InputText):
    emb = EMBEDDING_MODEL.encode(input.text)
    return {"embeddings": emb.tolist()}