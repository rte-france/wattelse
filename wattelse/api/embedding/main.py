#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pathlib import Path
import tomllib
from loguru import logger

from fastapi import FastAPI

from sentence_transformers import SentenceTransformer

from wattelse.api.embedding.routers import info, embeddings

# Load config file
CONFIG_FILE = Path(__file__).parent / "config.toml"
with open(CONFIG_FILE, "rb") as f:
    CONFIG = tomllib.load(f)

# Load embedding model
logger.debug(f"Loading embedding model : {CONFIG.model_name}")
EMBEDDING_MODEL = SentenceTransformer(CONFIG.model_name)

# Fix max model length error
if EMBEDDING_MODEL.max_seq_length == 514:
    EMBEDDING_MODEL.max_seq_length = 512


app = FastAPI()

app.include_router(info.router, tags=["Utils"])
app.include_router(embeddings.router, tags=["Embeddings"])
