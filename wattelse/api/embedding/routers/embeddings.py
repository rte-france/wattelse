from fastapi import APIRouter

from wattelse.api.embedding.main import EMBEDDING_MODEL
from wattelse.api.embedding.models import InputText


router = APIRouter()


@router.post("/encode")
def embed(input: InputText):
    emb = EMBEDDING_MODEL.encode(input.text, show_progress_bar=input.show_progress_bar)
    return {"embeddings": emb.tolist()}
