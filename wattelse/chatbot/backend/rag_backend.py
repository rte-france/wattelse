#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
import json
import logging
import os
from typing import List, Dict, BinaryIO

from fastapi.responses import StreamingResponse
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from loguru import logger

from wattelse.chatbot.backend import DATA_DIR
from wattelse.chatbot.backend.vector_database import format_docs, \
    load_document_collection

from wattelse.api.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION
from wattelse.chatbot.backend import retriever_config, generator_config, FASTCHAT_LLM, CHATGPT_LLM, OLLAMA_LLM, \
    LLM_CONFIGS, BM25, ENSEMBLE, MMR, SIMILARITY, SIMILARITY_SCORE_THRESHOLD
from wattelse.indexer.document_splitter import split_file
from wattelse.common.config_utils import parse_literal
from wattelse.indexer.document_parser import parse_file

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class RAGError(Exception):
    pass


def get_chat_model(llm_api_name) -> BaseChatModel:
    llm_config_file = LLM_CONFIGS.get(llm_api_name, None)
    if llm_config_file is None:
        raise RAGError(f"Unrecognized LLM API name {llm_api_name}")
    config = configparser.ConfigParser(converters={"literal": parse_literal})
    config.read(llm_config_file)
    api_config = config["API_CONFIG"]
    if llm_api_name == FASTCHAT_LLM:
        llm_config = {"openai_api_key": api_config["openai_api_key"],
                      "openai_api_base": api_config["openai_url"],
                      "model_name": api_config["model_name"],
                      "temperature": api_config["temperature"],
                      }
        return ChatOpenAI(**llm_config)
    elif llm_api_name == CHATGPT_LLM:
        llm_config = {"openai_api_key": api_config["openai_api_key"],
                      "model_name": api_config["model_name"],
                      "temperature": api_config["temperature"],
                      }
        return ChatOpenAI(**llm_config)
    elif llm_api_name == OLLAMA_LLM:
        llm_config = {"base_url": api_config["base_url"],
                      "model": api_config["model_name"],
                      "temperature": api_config["temperature"],
                      }
        # TODO: check if other parameters are needed
        return ChatOllama(**llm_config)
    else:
        raise RAGError(f"Unrecognized LLM API name {llm_api_name}")


class RAGBackEnd:
    def __init__(self, group: str):
        logger.debug(f"Initialization of chatbot backend for group {group}")

        # Load document collection
        self.document_collection = load_document_collection(group)

        # Retriever parameters
        self.top_n_extracts = retriever_config["top_n_extracts"]
        self.retrieval_method = retriever_config["retrieval_method"]
        self.similarity_threshold = retriever_config["similarity_threshold"]
        self.multi_query_mode = retriever_config["multi_query_mode"]

        # Generator parameters
        self.llm_api_name = generator_config["llm_api_name"]
        self.expected_answer_size = generator_config["expected_answer_size"]
        self.remember_recent_messages = generator_config["remember_recent_messages"]
        self.custom_prompt = generator_config["custom_prompt"]
        self.temperature = generator_config["temperature"]

        # Generate llm config for langchain
        self.llm = get_chat_model(self.llm_api_name)

    def add_file_to_collection(self, file_name: str, file: BinaryIO):
        """Add a file to the document collection"""
        # Store the file
        contents = file.read()
        path = DATA_DIR / self.document_collection.collection_name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(contents)
        logger.debug(f"File stored in: {path}")

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
                    f"File {filename} not present in the collection {self.document_collection.collection_name}, skippping removal")
                continue

            # remove info from vector database
            data = self.document_collection.collection.get(where={"file_name": filename}, include=["metadatas"])
            paths = list({meta["source"] for meta in data["metadatas"]})
            assert (len(paths) == 1)
            self.document_collection.collection.delete(self.document_collection.get_ids(filename))

            # remove file from disk
            os.remove(paths[0])
            logger.info(f"File {filename} removed from disk and vector database")

    def get_available_docs(self) -> List[str]:
        """Returns the list of documents in the collection"""
        data = self.document_collection.collection.get(include=["metadatas"])
        available_docs = list({d["file_name"] for d in data["metadatas"]})
        available_docs.sort()
        return available_docs

    def get_text_list(self, document_filter: Dict | None) -> List[str]:
        """Returns the list of texts in the collection, using the current document filter"""
        data = self.document_collection.collection.get(include=["documents"],
                                                       where={} if not document_filter else document_filter)
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

    def query_rag(self, message: str, history: List[dict[str, str]] = None, selected_files: List[str] = None, stream: bool = False) -> Dict | StreamingResponse:
        """Query the RAG"""
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")
        
        # Get document filter
        document_filter = self.get_document_filter(selected_files)

        # Configure retriever
        search_kwargs = {
            "k": self.top_n_extracts,  # number of retrieved docs
            "filter": {} if not document_filter else document_filter,
        }
        if self.retrieval_method == SIMILARITY_SCORE_THRESHOLD:
            search_kwargs["score_threshold"] = self.similarity_threshold

        if self.retrieval_method in [MMR, SIMILARITY, SIMILARITY_SCORE_THRESHOLD]:
            dense_retriever = self.document_collection.collection.as_retriever(
                search_type=self.retrieval_method,
                search_kwargs=search_kwargs
            )
            retriever = dense_retriever

        elif self.retrieval_method in [BM25, ENSEMBLE]:
            bm25_retriever = BM25Retriever.from_texts(self.get_text_list(document_filter))
            bm25_retriever.k = self.top_n_extracts
            if self.similarity_threshold == BM25:
                retriever = bm25_retriever
            else:  # ENSEMBLE
                dense_retriever = self.document_collection.collection.as_retriever(
                    search_type=MMR,
                    search_kwargs=search_kwargs
                )
                retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever])

        if self.multi_query_mode:
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=self.llm
            )

        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(input_variables=['context', 'expected_answer_size', 'query'],
                                    messages=[HumanMessagePromptTemplate(
                                        prompt=PromptTemplate(
                                            input_variables=['context', 'expected_answer_size', 'query'],
                                            template=self.custom_prompt)
                                    )])

        # - RAG chain
        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )
        # returns both answer and sources
        rag_chain = RunnableParallel(
            {"context": retriever if not self.multi_query_mode else multi_query_retriever,
             "expected_answer_size": self.get_detail_level, "query": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # TODO: implement reranking (optional)

        # Handle conversation history
        contextualized_question = self.contextualize_question(message, history)
        logger.debug(f"Calling RAG chain for question : \"{contextualized_question}\"...")

        # Handle answer
        chunks = []
        relevant_extracts = []
        if stream:
            # stream response on server side...
            # TODO: manage streaming to the client
            for chunk in rag_chain.stream(contextualized_question):
                s = chunk.get("context")
                if s:
                    sources = s
                    # Transform sources
                    relevant_extracts = [{"content": s.page_content, "metadata": s.metadata} for s in sources]
                chunks.append(chunk.get("answer", ""))
                print(chunk, end="", flush=True)
            answer = "".join(chunks)
            # return StreamingResponse(streamer(rag_chain.stream(contextualized_question)))
        else:
            resp = rag_chain.invoke(contextualized_question)
            answer = resp.get("answer")
            sources = resp.get("context")
            # Transform sources
            relevant_extracts = [{"content": s.page_content, "metadata": s.metadata} for s in sources]

        # Return answer and sources
        return {"answer": answer, "relevant_extracts": relevant_extracts}

    def contextualize_question(self, message: str, history: List[dict[str, str]] = None) -> str:
        """
        If self.remember_recent_messages is False or no message in history:
            Return last user query
        Else :
            Use recent interaction context to enrich the user query
        """
        if not self.remember_recent_messages or history is None:
            return message
        else:
            logger.debug("Contextualizing prompt with history...")
            prompt = ChatPromptTemplate(input_variables=["history", "query"],
                                        messages=[HumanMessagePromptTemplate(
                                            prompt=PromptTemplate(
                                                input_variables=["history", "query"],
                                                template=FR_USER_MULTITURN_QUESTION_SPECIFICATION)
                                        )])

            chain = ({"query": RunnablePassthrough(), "history": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser())
            
            # Format messages into a single string
            history_as_text = ""
            for turn in history:
                history_as_text += f"{turn['role']}: {turn['content']}\n"
            contextualized_question = chain.invoke([history_as_text, message])
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
