import configparser
import json
import os
from datetime import datetime
from typing import List

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_openai import ChatOpenAI
from loguru import logger

from wattelse.chatbot import DATA_DIR
from wattelse.chatbot.backend.data_management.vector_database import format_docs, \
    load_document_collection

from wattelse.api.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION
from wattelse.chatbot.backend import retriever_config, generator_config, FASTCHAT_LLM, CHATGPT_LLM, OLLAMA_LLM, \
    LLM_CONFIGS
from wattelse.chatbot.chat_history import ChatHistory
from wattelse.common.config_utils import parse_literal
from wattelse.indexer.document_parser import parse_file


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
    def __init__(self, login):
        logger.debug(f"Initialization of chatbot backend for user {login}")
        # Initialize history
        log_chat_history_on_disk = True
        if log_chat_history_on_disk:
            self.chat_history = ChatHistory(DATA_DIR / "chat_history" / login /
                                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            self.chat_history = ChatHistory()

        # Load document collection
        self.document_collection = load_document_collection(login)
        self.document_filter = None

        # Retriever parameters
        self.top_n_extracts = retriever_config["top_n_extracts"]
        self.retrieval_method = retriever_config["retrieval_method"]
        self.similarity_threshold = retriever_config["similarity_threshold"]

        # Generator parameters
        self.llm_api_name = generator_config["llm_api_name"]
        self.expected_answer_size = generator_config["expected_answer_size"]
        self.remember_recent_messages = generator_config["remember_recent_messages"]
        self.custom_prompt = generator_config["custom_prompt"]
        self.temperature = generator_config["temperature"]

        # Generate llm config for langchain
        self.llm = get_chat_model(self.llm_api_name)

    def add_file_to_collection(self, file: UploadFile):
        """Add a file to the document collection"""
        # Store the file
        contents = file.file.read()
        path = DATA_DIR / self.document_collection.collection_name / file.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(contents)
        logger.debug(f"File stored in: {path}")

        # Parse file
        logger.debug(f"Parsing: {path}")
        docs = parse_file(path)

        # Split the file into smaller chunks as a list of Document
        logger.debug(f"Chunking: {path}")
        splits = self.split_file(path.suffix, docs)
        logger.info(f"Number of chunks for file {file.filename}: {len(splits)}")

        # Store and embed documents in the vector database
        self.document_collection.add_documents(splits)

    def split_file(self, file_extension: str, docs: List[Document]):
        """Split a file into smaller chunks - the chunking method depends on file type"""
        if file_extension == ".md":
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ])
        elif file_extension in [".htm", ".html"]:
            text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ])
        else:
            # TODO: check split parameters
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        return splits

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

    def select_docs(self, file_names: List[str]):
        """Create a filter on the document collection based on a list of file names"""
        self.document_filter = {"file_name": " $or ".join(file_names)} if file_names else None

    def select_by_keywords(self, keywords: List[str]):
        """Create a filter on the document collection based on a list of keywords"""
        # TODO: to be implemented
        self.document_filter = None

    def query_rag(self, question: str, **kwargs) -> str | StreamingResponse:
        """Query the RAG"""
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")

        # Configure retriever
        retriever = self.document_collection.collection.as_retriever(
            search_type=self.retrieval_method,
            search_kwargs={
                "k": self.top_n_extracts,  # number of retrieved docs
                "filter": {} if not self.document_filter else self.document_filter,
                "score_threshold": self.similarity_threshold
            })
        # docs = retriever.get_relevant_documents(question)
        # logger.debug(f'Number of retrieved docs = {len(docs)}')
        # logger.debug(f"Relevant documents: {docs}")

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
            {"context": retriever, "expected_answer_size": self.get_detail_level, "query": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # TODO: implement reranking (optional)
        logger.debug(f"Calling RAG chain for question {question}...")

        # Handle conversation history
        contextualized_question = self.contextualize_question(question) if self.remember_recent_messages else question

        # Handle answer
        stream = True
        chunks = []
        sources = []
        if stream:
            # stream response on server side...
            # TODO: manage streaming to the client
            for chunk in rag_chain.stream(contextualized_question):
                s = chunk.get("context")
                if s:
                    sources = s
                chunks.append(chunk.get("answer", ""))
                print(chunk, end="", flush=True)
            answer = "".join(chunks)

            # return StreamingResponse(streamer(rag_chain.stream(contextualized_question)))
        else:
            resp = rag_chain.invoke(contextualized_question)
            answer = resp.get("answer")
            sources = resp.get("context")

        # Update chat history
        self.chat_history.add_to_database(question, answer)

        # Return answer and sources
        #TODO: fix output format
        return answer #+ "\n" + str(sources)

    def contextualize_question(self, question: str) -> str:
        """Use recent interaction context to enrich the user query"""
        logger.debug("Contextualizing prompt with history...")
        recent_history = self.chat_history.get_recent_history()
        if not recent_history:
            logger.warning("No recent history available!")
            return question
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
        contextualized_question = chain.invoke([question, recent_history])
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
