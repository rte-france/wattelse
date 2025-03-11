from fastapi import APIRouter

from wattelse.api.embedding.config.settings import get_config

# Load the configuration
CONFIG = get_config()


router = APIRouter()


@router.get("/model_name")
def get_model_name():
    return CONFIG.model_name


@router.get("/num_workers")
def get_num_workers():
    return CONFIG.number_workers
