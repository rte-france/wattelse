from typing import Annotated

from fastapi import APIRouter, Security, Depends
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from sentence_transformers import SentenceTransformer

from wattelse.api.embedding.config.settings import get_config
from wattelse.api.embedding.models import InputText
from wattelse.api.security import (
    TokenData,
    get_current_client,
    FULL_ACCESS,
    RESTRICTED,
    Token,
    get_token,
)

# Load the configuration
CONFIG = get_config()

# Load embedding model
logger.debug(f"Loading embedding model : {CONFIG.model_name}")
EMBEDDING_MODEL = SentenceTransformer(CONFIG.model_name)

# Fix max model length error
if EMBEDDING_MODEL.max_seq_length == 514:
    EMBEDDING_MODEL.max_seq_length = 512
router = APIRouter()


@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    return get_token(form_data)


@router.post("/encode", summary="Embed data (requires full_access scope)")
def embed(
    input: InputText,
    current_client: TokenData = Security(get_current_client, scopes=[FULL_ACCESS]),
):
    emb = EMBEDDING_MODEL.encode(input.text, show_progress_bar=input.show_progress_bar)
    return {"embeddings": emb.tolist()}
