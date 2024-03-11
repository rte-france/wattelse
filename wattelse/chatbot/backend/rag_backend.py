import os
from datetime import datetime
from typing import List

from fastapi import UploadFile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from loguru import logger

from wattelse.chatbot import (
    TEMPERATURE, DATA_DIR,
)
from wattelse.chatbot.backend.data_management.vector_database import DocumentCollection, format_docs
from wattelse.chatbot.backend.user_management.user_manager import get_document_collections_for_user

from wattelse.api.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION, FR_USER_BASE_RAG
from wattelse.chatbot.chat_history import ChatHistory
from wattelse.indexer.document_parser import parse_file


class RAGError(Exception):
    pass


class RAGBackEnd:
    def __init__(self, login, **kwargs):
        logger.debug(f"Initialization of chatbot backend for user {login}")
        # TODO: parameter management
        self._retrieval_mode = kwargs.get("retrieval_mode")

        # Load document collection
        self._document_collection = self.load_document_collection(login)
        self._document_filter = None

        # TODO: parameter management
        self._llm_config = {"openai_api_key": "EMPTY",
                            "openai_api_base": "http://localhost:8888/v1",
                            "model_name": "bofenghuang/vigostral-7b-chat",
                            "temperature": TEMPERATURE
                            }
        self.llm = ChatOpenAI(**self._llm_config)

        # Initialize history
        # TODO: parameter management
        log_chat_history = True
        if log_chat_history:
            self._chat_history = ChatHistory(DATA_DIR / "chat_history" / login /
                                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            self._chat_history = ChatHistory()

    @property
    def document_collection(self):
        return self._document_collection

    @property
    def chat_history(self):
        return self._chat_history


    def load_document_collection(self, login) -> DocumentCollection:
        """Retrieves the document collection the user can access to"""
        # TODO: improve in case of user has access to multiple collections
        user_collections_names = get_document_collections_for_user(login)
        logger.debug(f"DocumentCollections for user {login}: {user_collections_names}")
        if not user_collections_names:
            raise RAGError(f"No document collection for user {login}")
        return DocumentCollection(user_collections_names[0])

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
        # TODO: config splitter
        logger.debug(f"Chunking: {path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        logger.info(f"Number of chunks for file {file.filename}: {len(splits)}")

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

    def select_docs(self, file_names: List[str]):
        """Create a filter on the document collection based on a list of file names"""
        self._document_filter = {"file_name": " $or ".join(file_names)} if file_names else None

    def query_rag(self, question: str, history=None, **kwargs) -> str:
        """Query the RAG"""
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")

        # Configure retriever
        # TODO: improve parameter config
        retriever = self.document_collection.collection.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,  # number of retrieved docs
                "filter": {} if not self._document_filter else self._document_filter,
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
                                            template=FR_USER_BASE_RAG)
                                    )])

        # - RAG chain
        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )
        # returns both answer and sources
        # TODO: chain with history
        rag_chain = RunnableParallel(
            {"context": retriever, "expected_answer_size": get_detail_level, "query": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # TODO: implement reranking (optional)
        logger.debug(f"Calling RAG chain for question {question}...")

        # Handle conversation history
        # TODO - handle parameter: remember_recent_messages
        remember_recent_messages = True
        contextualized_question = question if not remember_recent_messages else self.contextualize_question(question)

        # Handle answer
        stream = True
        chunks = []
        sources = []
        if stream:
            # stream response on server side...
            for chunk in rag_chain.stream(contextualized_question):
                s = chunk.get("context")
                if s:
                    sources = s
                chunks.append(chunk.get("answer", ""))
                print(chunk, end="", flush=True)
            answer = "".join(chunks)
        else:
            resp = rag_chain.invoke(contextualized_question)
            answer = resp.get("answer")
            sources = resp.get("context")

        # Update chat history
        self.chat_history.add_to_database(question, answer)

        # Return answer and sources
        return answer + "\n" + str(sources)

    def contextualize_question(self, question: str) -> str:
        """Use recent interaction context to enrich the user query"""
        logger.debug("Contextualizing prompt with history...")
        recent_history = self.chat_history.get_recent_history()
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


def get_detail_level(question: str):
    """Returns the level of detail we wish in the answer. Values are in this range: {"courte", "détaillée"}"""
    return "courte"

