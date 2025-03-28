#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import logging
import os
import shutil
from pathlib import Path
from typing import BinaryIO, List, Dict, Union

from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from loguru import logger
from starlette.responses import StreamingResponse

from wattelse.rag_backend import DATA_DIR
from wattelse.api.embedding.client import EmbeddingAPIClient
from wattelse.rag_backend.vector_database import (
    DocumentCollection,
    format_docs,
)

from wattelse.rag_backend.configs.settings import RAGBackendConfig
from wattelse.rag_backend import (
    BM25,
    ENSEMBLE,
    MMR,
    SIMILARITY,
    SIMILARITY_SCORE_THRESHOLD,
)
from wattelse.rag_backend.indexer.document_splitter import split_file
from wattelse.rag_backend.indexer.document_parser import parse_file

from wattelse.rag_backend.utils import (
    RAGError,
    get_chat_model,
    preprocess_streaming_data,
    filter_history,
    get_history_as_text,
)

from wattelse.rag_backend.configs import CONFIG_NAME_TO_CONFIG_PATH

from wattelse.common.config_utils import load_toml_config

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Load one config key as a default_config name"
default_config = next(iter(CONFIG_NAME_TO_CONFIG_PATH))


class RAGBackend:
    def __init__(self, group_id: str, config: str | dict | Path = default_config):
        """
        Creates a RAGBackend for a given `group_id`.
        3 ways to set the configuration depending on `config` type:
            - str: default config name (see `wattelse/chatbot/backend/configs`)
            - dict: config dict that should follow format found in default config files
            - Path: path to a .toml config file
        """
        self.group_id = group_id
        self.config = self._load_config(config)
        self.llm = get_chat_model(self.config.generator)
        self.embedding_function = EmbeddingAPIClient(
            self.config.retriever.embedding_api_url
        )
        self.document_collection = DocumentCollection(
            self.group_id,
            self.embedding_function,
        )

    def _load_config(self, config: str | dict | Path) -> RAGBackendConfig:
        """
        Function that implements RAGBackend configuration logic.
        """
        # If `config` is str, return associated default config
        if isinstance(config, str):
            if config not in CONFIG_NAME_TO_CONFIG_PATH:
                raise Exception(
                    f"config_name '{config}' not found. Please use one of the following: {list(CONFIG_NAME_TO_CONFIG_PATH.keys())}"
                )
            config_file_path = CONFIG_NAME_TO_CONFIG_PATH[config]
            config = load_toml_config(config_file_path)
            return RAGBackendConfig(**config)

        # If `config` is dict, return it directly
        elif isinstance(config, dict):
            return RAGBackendConfig(config)

        # If `config` is Path, load config from toml file
        elif isinstance(config, Path):
            config = load_toml_config(config)
            return RAGBackendConfig(**config)

        # Else raise TypeError
        else:
            raise TypeError(
                f"Invalid input type for 'config': {type(config).__name__}. "
                "Expected one of: str, dict, or pathlib.Path."
            )

    def get_llm_model_name(self):
        return self.config.generator.openai_default_model

    def add_file_to_collection(self, file_name: str, file: BinaryIO):
        """Add a file to the document collection"""
        # Store the file
        contents = file.read()
        path = DATA_DIR / self.document_collection.collection_name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(contents)
        logger.debug(f"File {file_name} stored in: {path}")

        # Parse file
        logger.debug(f"Parsing: {path}")
        docs = parse_file(path)

        # Split the file into smaller chunks as a list of Document
        logger.debug(f"Chunking: {path}")
        splits = split_file(path.suffix, docs)
        logger.info(f"Number of chunks for file {file_name}: {len(splits)}")

        # Store and embed documents in the vector database
        self.document_collection.add_documents(splits)

    def remove_docs(self, doc_file_names: List[str]):
        """Remove a list of documents from the document collection"""
        for filename in doc_file_names:
            # check if the file is already in the document collection
            if not self.document_collection.is_present(filename):
                logger.warning(
                    f"File {filename} not present in the collection {self.document_collection.collection_name}, skippping removal"
                )
                continue

            # remove info from vector database
            data = self.document_collection.collection.get(
                where={"file_name": filename}, include=["metadatas"]
            )
            paths = list({meta["source"] for meta in data["metadatas"]})
            assert len(paths) == 1
            self.document_collection.collection.delete(
                self.document_collection.get_ids(filename)
            )

            # remove file from disk
            os.remove(paths[0])
            logger.info(f"File {filename} removed from disk and vector database")

    def clear_collection(self):
        """
        Delete all documents and embeddings from the document collection.
        WARNING: all files will be lost permanently. You should use this function with caution.
        """
        logger.warning(
            f"Clearing document collection: {self.document_collection.collection_name}"
        )

        # Delete ChromaDB collection
        try:
            self.document_collection.client.delete_collection(
                self.document_collection.collection_name
            )
        except ValueError:
            logger.warning(
                f"Collection {self.document_collection.collection_name} not found"
            )

        # Delete collection files folder
        collection_path = DATA_DIR / self.document_collection.collection_name
        try:
            shutil.rmtree(str(collection_path))
        except FileNotFoundError:
            logger.warning(f"No documents found at {collection_path}")

    def get_available_docs(self) -> List[str]:
        """Returns the list of documents in the collection"""
        data = self.document_collection.collection.get(include=["metadatas"])
        available_docs = list({d["file_name"] for d in data["metadatas"]})
        available_docs.sort()
        return available_docs

    def get_file_path(self, file_name: str) -> str:
        """Returns the contents of a file of the collection"""
        file_path = DATA_DIR / self.document_collection.collection_name / file_name
        if file_path.is_file():
            return file_path
        else:
            return None

    def get_doc_list(self, document_filter: Dict | None) -> list[Document]:
        """Returns the list of documents in the collection, using the current document filter"""
        data = self.document_collection.collection.get(
            include=["documents", "metadatas"],
            where={} if not document_filter else document_filter,
        )
        langchain_documents = []
        for doc, meta in zip(data["documents"], data["metadatas"]):
            langchain_doc = Document(page_content=doc, metadata=meta)
            langchain_documents.append(langchain_doc)
        return langchain_documents

    def get_text_list(self, document_filter: Dict | None) -> List[str]:
        """Returns the list of texts in the collection, using the current document filter"""
        data = self.document_collection.collection.get(
            include=["documents", "metadatas"],
            where={} if not document_filter else document_filter,
        )
        return data["documents"]

    def get_document_filter(self, file_names: List[str]):
        """Create a filter on the document collection based on a list of file names"""
        if not file_names:
            return None
        elif len(file_names) == 1:
            return {"file_name": file_names[0]}
        else:
            return {"$or": [{"file_name": f} for f in file_names]}

    def select_by_keywords(self, keywords: List[str]):
        """Create a filter on the document collection based on a list of keywords"""
        # TODO: to be implemented
        pass

    def query_rag(
        self,
        message: str,
        history: List[dict[str, str]] = None,
        group_system_prompt: str = None,
        selected_files: List[str] = None,
        stream: bool = False,
    ) -> Union[Dict, StreamingResponse]:
        """Query the RAG"""
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")

        # Raise error if query is empty or not str
        if not isinstance(message, str):
            raise ValueError("RAGBackend.query_rag: query is not a string")
        if message == "":
            raise ValueError("RAGBackend.query_rag(): query is empty")

        # Get document filter
        document_filter = self.get_document_filter(selected_files)

        # Configure retriever
        search_kwargs = {
            "k": self.config.retriever.top_n_extracts,  # number of retrieved docs
            "filter": {} if not document_filter else document_filter,
        }
        if self.config.retriever.retrieval_method == SIMILARITY_SCORE_THRESHOLD:
            search_kwargs["score_threshold"] = (
                self.config.retriever.similarity_threshold
            )

        if self.config.retriever.retrieval_method in [
            MMR,
            SIMILARITY,
            SIMILARITY_SCORE_THRESHOLD,
        ]:
            dense_retriever = self.document_collection.collection.as_retriever(
                search_type=self.config.retriever.retrieval_method,
                search_kwargs=search_kwargs,
            )
            retriever = dense_retriever

        elif self.config.retriever.retrieval_method in [BM25, ENSEMBLE]:
            bm25_retriever = BM25Retriever.from_documents(
                self.get_doc_list(document_filter)
            )
            bm25_retriever.k = self.config.retriever.top_n_extracts
            if self.config.retriever.retrieval_method == BM25:
                retriever = bm25_retriever
            else:  # ENSEMBLE
                dense_retriever = self.document_collection.collection.as_retriever(
                    search_type=MMR, search_kwargs=search_kwargs
                )
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, dense_retriever]
                )

        if self.config.retriever.multi_query_mode:
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=self.llm
            )

        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(
            input_variables=["context", "history", "query", "group_system_prompt"],
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["group_system_prompt"],
                        template=self.config.generator.system_prompt,
                    )
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context", "history", "query"],
                        template=self.config.generator.user_prompt,
                    )
                ),
            ],
        )

        # - RAG chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )
        # returns both answer and sources
        rag_chain = RunnableParallel(
            {
                "context": (
                    retriever
                    if not self.config.retriever.multi_query_mode
                    else multi_query_retriever
                ),
                "history": (lambda _: get_history_as_text(history)),
                "query": (lambda _: message),
                "contextualized_query": RunnablePassthrough(),
                "group_system_prompt": (lambda _: group_system_prompt),
            }
        ).assign(answer=rag_chain_from_docs)

        # TODO: implement reranking (optional)

        # Handle conversation history
        contextualized_question = "query : " + self.contextualize_question(
            message, history
        )
        logger.debug(f'Calling RAG chain for question : "{message}"...')

        # Handle answer
        if stream:
            return preprocess_streaming_data(rag_chain.stream(contextualized_question))
        else:
            resp = rag_chain.invoke(contextualized_question)
            answer = resp.get("answer")
            sources = resp.get("context")
            # Transform sources
            relevant_extracts = [
                {"content": s.page_content, "metadata": s.metadata} for s in sources
            ]

            # Return answer and sources
            return {"answer": answer, "relevant_extracts": relevant_extracts}

    def contextualize_question(
        self,
        message: str,
        history: List[dict[str, str]] = None,
        interaction_window: int = 3,
    ) -> str:
        """
        If self.remember_recent_messages is False or no message in history:
            Return last user query
        Else :
            Use recent interaction context to enrich the user query
        """
        if not self.config.generator.remember_recent_messages or history is None:
            return message
        else:
            history = filter_history(history, interaction_window)

            logger.debug("Contextualizing prompt with history...")
            prompt = ChatPromptTemplate(
                input_variables=["history", "query"],
                messages=[
                    SystemMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=[],
                            template=self.config.generator.system_prompt_query_contextualization,
                        )
                    ),
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=["history", "query"],
                            template=self.config.generator.user_prompt_query_contextualization,
                        )
                    ),
                ],
            )

            chain = prompt | self.llm | StrOutputParser()

            # Format messages into a single string
            history_as_text = get_history_as_text(history)
            contextualized_question = chain.invoke(
                {"query": message, "history": history_as_text}
            )
            logger.debug(f"Contextualized question: {contextualized_question}")
            return contextualized_question
