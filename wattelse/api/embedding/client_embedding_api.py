#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
from pathlib import Path
import requests
import json

from click.testing import EchoingStdin
from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from loguru import logger
from typing import List

import numpy as np

BATCH_DOCUMENT_SIZE = 5000
MAX_DOCS_PER_REQUEST = 50000

class EmbeddingAPI(Embeddings):
    """
    Custom Embedding API client, can integrate seamlessly with langchain
    """

    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "embedding_api.cfg")
        self.port = config.get("EMBEDDING_API_CONFIG", "port")
        self.host = config.get("EMBEDDING_API_CONFIG", "host")
        self.url = f"http://{self.host}:{self.port}"
        self.model_name = config.get("EMBEDDING_API_CONFIG", "model_name")

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        return self.model_name

    def embed_query(
        self, text: str | List[str], show_progress_bar: bool = False
    ) -> List[float]:
        if type(text) == str:
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": text, "show_progress_bar": show_progress_bar}),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done")
            return embeddings.tolist()[0]
        else:
            logger.error(f"Error: {response.status_code}")

    def embed_batch(
        self, texts: List[str], show_progress_bar: bool = True
    ) -> List[List[float]]:
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": texts, "show_progress_bar": show_progress_bar}),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done for batch")
            return embeddings.tolist()
        else:
            logger.error(f"Error: {response.status_code}")
            return []

    def embed_documents(
        self, texts: List[str], show_progress_bar: bool = True, batch_size: int = BATCH_DOCUMENT_SIZE
    ) -> List[List[float]]:
        if len(texts) > MAX_DOCS_PER_REQUEST:
            # Too many documents to embed in one request, refuse it
            logger.error(f"Error: Too many documents to be embedded ({len(texts)} chunks)")
            raise ValueError(f"Error: Too many documents to be embedded ({len(texts)} chunks)")

        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")

        # Split texts into chunks
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.debug(f"Computing embeddings on {len(texts)} documents using {batches} batches...")

        # Parallel request
        results = Parallel(n_jobs=-1)(delayed(self.embed_batch)(batch, show_progress_bar) for batch in batches)

        # Check results
        if any(result == [] for result in results):
            raise ValueError("At least one batch processing failed. Documents are not embedded.")

        # Compile results
        embeddings = [embedding for result in results for embedding in result]
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        # FIXME!
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # FIXME!
        return self.embed_documents(texts)
