#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import logging
import os
import shutil
from typing import BinaryIO, List, Dict, Union

from langchain.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from loguru import logger
from starlette.responses import StreamingResponse

from wattelse.chatbot.backend import DATA_DIR
from wattelse.chatbot.backend.vector_database import (
    format_docs,
    load_document_collection,
)

from wattelse.api.prompts import (
    FR_SYSTEM_RAG_LLAMA3,
    FR_USER_RAG_LLAMA3,
    FR_SYSTEM_QUERY_CONTEXTUALIZATION_LLAMA3,
    FR_USER_QUERY_CONTEXTUALIZATION_LLAMA3,
)
from wattelse.chatbot.backend import (
    retriever_config,
    generator_config,
    BM25,
    ENSEMBLE,
    MMR,
    SIMILARITY,
    SIMILARITY_SCORE_THRESHOLD,
)
from wattelse.indexer.document_splitter import split_file
from wattelse.indexer.document_parser import parse_file

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

DEFAULT_TEMPERATURE = 0.1
AZURE_API_VERSION = "2024-02-01"


class RAGError(Exception):
    pass


def get_chat_model() -> BaseChatModel:
    endpoint = generator_config["openai_endpoint"]
    if "azure.com" not in endpoint:
        llm_config = {
            "openai_api_key": generator_config["openai_api_key"],
            "openai_api_base": generator_config["openai_endpoint"],
            "model_name": generator_config["openai_default_model"],
            "temperature": DEFAULT_TEMPERATURE,
        }
        return ChatOpenAI(**llm_config)
    else:
        llm_config = {
            "openai_api_key": generator_config["openai_api_key"],
            "azure_endpoint": generator_config["openai_endpoint"],
            "azure_deployment": generator_config["openai_default_model"],
            "api_version": AZURE_API_VERSION,
            "temperature": DEFAULT_TEMPERATURE,
        }
        llm = AzureChatOpenAI(**llm_config)
        llm.model_name = generator_config[
            "openai_default_model"
        ]  # with Azure, set to gpt3.5-turbo by default
        return llm


def preprocess_streaming_data(streaming_data):
    """Generator to preprocess the streaming data coming from LangChain `rag_chain.stream()`.
    First sent chunk contains relevant_extracts in a convenient format.
    Following chunks contain the actual response from the model token by token.
    """
    for chunk in streaming_data:
        context_chunk = chunk.get("context", None)
        if context_chunk is not None:
            relevant_extracts = [
                {"content": s.page_content, "metadata": s.metadata}
                for s in context_chunk
            ]
            relevant_extracts = {"relevant_extracts": relevant_extracts}
            yield json.dumps(relevant_extracts)
        answer_chunk = chunk.get("answer")
        if answer_chunk:
            yield json.dumps(chunk)


def filter_history(history, window_size):
    # window size = question + answser, we return the last ones
    return history[-2 * window_size :]


def get_document_filter(file_names: List[str]):
    """Create a filter on the document collection based on a list of file names"""
    if not file_names:
        return None
    elif len(file_names) == 1:
        return {"file_name": file_names[0]}
    else:
        return {"$or": [{"file_name": f} for f in file_names]}


class RAGBackEnd:
    def __init__(self, group_id: str):
        logger.info(f"[Group: {group_id}] Initialization of chatbot backend")

        # Load document collection
        self.document_collection = load_document_collection(group_id)

        # Retriever parameters
        self.top_n_extracts = retriever_config["top_n_extracts"]
        self.retrieval_method = retriever_config["retrieval_method"]
        self.similarity_threshold = retriever_config["similarity_threshold"]
        self.multi_query_mode = retriever_config["multi_query_mode"]
        self.use_context_compression = retriever_config["use_context_compression"]

        # Generator parameters
        self.expected_answer_size = generator_config["expected_answer_size"]
        self.remember_recent_messages = generator_config["remember_recent_messages"]
        self.temperature = generator_config["temperature"]
        # Generate llm config for langchain
        self.llm = get_chat_model()

        # Prompts
        self.system_prompt = FR_SYSTEM_RAG_LLAMA3
        self.user_prompt = FR_USER_RAG_LLAMA3
        self.system_prompt_query_contextualization = (
            FR_SYSTEM_QUERY_CONTEXTUALIZATION_LLAMA3
        )
        self.user_prompt_query_contextualization = (
            FR_USER_QUERY_CONTEXTUALIZATION_LLAMA3
        )

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

    def select_by_keywords(self, keywords: List[str]):
        """Create a filter on the document collection based on a list of keywords"""
        # TODO: to be implemented
        pass

    def query_rag(
        self,
        message: str,
        history: List[dict[str, str]] = None,
        selected_files: List[str] = None,
        stream: bool = False,
    ) -> Union[Dict, StreamingResponse]:
        """Query the RAG"""
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")

        # Get document filter
        document_filter = get_document_filter(selected_files)

        # Configure retriever
        search_kwargs = {
            "k": self.top_n_extracts,  # number of retrieved docs
            "filter": {} if not document_filter else document_filter,
        }
        if self.retrieval_method == SIMILARITY_SCORE_THRESHOLD:
            search_kwargs["score_threshold"] = self.similarity_threshold

        if self.retrieval_method in [MMR, SIMILARITY, SIMILARITY_SCORE_THRESHOLD]:
            dense_retriever = self.document_collection.collection.as_retriever(
                search_type=self.retrieval_method, search_kwargs=search_kwargs
            )
            retriever = dense_retriever

        elif self.retrieval_method in [BM25, ENSEMBLE]:
            bm25_retriever = BM25Retriever.from_documents(
                self.get_doc_list(document_filter)
            )
            bm25_retriever.k = self.top_n_extracts
            if self.retrieval_method == BM25:
                retriever = bm25_retriever
            else:  # ENSEMBLE
                dense_retriever = self.document_collection.collection.as_retriever(
                    search_type=MMR, search_kwargs=search_kwargs
                )
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, dense_retriever]
                )

        selected_retriever = retriever

        # Retriever for multi-query mode
        if self.multi_query_mode:
            selected_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=self.llm
            )

        # Retriever for context compression
        if self.use_context_compression:
            # contextual compression
            compressor = LLMChainExtractor.from_llm(self.llm)
            selected_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(
            input_variables=["context", "history", "query"],
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=[], template=self.system_prompt
                    )
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context", "history", "query"],
                        template=self.user_prompt,
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
                "context": selected_retriever,
                "history": (lambda _: get_history_as_text(history)),
                "query": (lambda _: message),
                "contextualized_query": RunnablePassthrough(),
            }
        ).assign(answer=rag_chain_from_docs)

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
        if not self.remember_recent_messages or history is None:
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
                            template=self.system_prompt_query_contextualization,
                        )
                    ),
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=["history", "query"],
                            template=self.user_prompt_query_contextualization,
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

    def get_detail_level(self, question: str):
        """Returns the level of detail we wish in the answer. Values are in this range: {"courte", "détaillée"}"""
        return "courte" if self.expected_answer_size == "short" else "détaillée"


def streamer(stream):
    for chunk in stream:
        if chunk.get("context"):
            yield ""
        else:
            yield json.dumps(chunk) + "\n"


def get_history_as_text(history: List[dict[str, str]]) -> str:
    """Format conversation history as a text string"""
    history_as_text = ""
    if history is not None:
        for turn in history:
            history_as_text += f"{turn['role']}: {turn['content']}\n"
    return history_as_text
