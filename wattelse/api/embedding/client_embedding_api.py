#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
from pathlib import Path
import requests
import json

from langchain_core.embeddings import Embeddings
from loguru import logger
from typing import List

import numpy as np

class EmbeddingAPI(Embeddings):
    """
    Custom Embedding API client, can integrate seamlessly with langchain
    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "embedding_api.cfg")
        self.port = config.get("EMBEDDING_API_CONFIG", "port")
        self.url = f'http://localhost:{self.port}'
        self.model_name = config.get("EMBEDDING_API_CONFIG", "model_name")

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        return self.model_name
    
    def embed_query(self, text: str | List[str], show_progress_bar: bool = False) -> List[float]:
        if type(text)==str:
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug(f"Computing embeddings...")
        response = requests.post(self.url+'/encode', data=json.dumps({'text': text, 'show_progress_bar': show_progress_bar}))
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done")
            return embeddings.tolist()[0]
        else:
            logger.error(f"Error: {response.status_code}")

    def embed_documents(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug(f"Computing embeddings...")
        response = requests.post(self.url+'/encode', data=json.dumps({'text': texts, 'show_progress_bar': show_progress_bar}))
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done")
            return embeddings.tolist()
        else:
            logger.error(f"Error: {response.status_code}")

    async def aembed_query(self, text: str) -> List[float]:
        #FIXME!
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        #FIXME!
        return self.embed_documents(texts)