from fastapi import APIRouter

from wattelse.api.embedding.main import CONFIG


router = APIRouter()


@router.get("/model_name")
def get_model_name():
    return CONFIG.model_name


@router.get("/num_workers")
def get_num_workers():
    return CONFIG.num_workers
