import os
from functools import lru_cache
from typing import List

from fastapi import UploadFile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from loguru import logger
from sentence_transformers import CrossEncoder

from wattelse.chatbot import (
    MAX_TOKENS,
    RETRIEVAL_HYBRID_RERANKER,
    RETRIEVAL_BM25,
    RETRIEVAL_HYBRID,
    FASTCHAT_LLM,
    OLLAMA_LLM,
    CHATGPT_LLM,
    TEMPERATURE, DATA_DIR,
)
from wattelse.chatbot.backend.data_management.vector_database import DocumentCollection, format_docs
from wattelse.chatbot.backend.user_management.user_manager import get_document_collections_for_user

from wattelse.api.prompts import FR_USER_MULTITURN_QUESTION_SPECIFICATION, FR_USER_BASE_RAG
from wattelse.api.fastchat.client_fastchat_api import FastchatAPI
from wattelse.api.ollama.client_ollama_api import OllamaAPI
from wattelse.api.openai.client_openai_api import OpenAI_API
from wattelse.api.embedding.client_embedding_api import EmbeddingAPI
from wattelse.indexer.document_parser import parse_file


@lru_cache(maxsize=3)
def initialize_embedding_api_client():
    """Load EmbeddingAPI"""
    logger.info(f"Initializing EmbeddingAPI")
    embedding_api = EmbeddingAPI()
    return embedding_api


@lru_cache(maxsize=2)
def initialize_reranker_model(reranker_model_name: str):
    """Load reranker_model"""
    logger.info(f"Initializing reranker model: {reranker_model_name}")
    reranker_model = CrossEncoder(reranker_model_name)
    return reranker_model


@lru_cache(maxsize=3)
def initialize_llm_api_client(llm_api_name: str):
    logger.info(f"Initializing LLM API: {llm_api_name}")
    if llm_api_name == FASTCHAT_LLM:
        return FastchatAPI()
    elif llm_api_name == OLLAMA_LLM:
        return OllamaAPI()
    elif llm_api_name == CHATGPT_LLM:
        return OpenAI_API()
    else:
        logger.error(f"Unknow API name : {llm_api_name}")


def enrich_query(llm_api, query: str, history):
    """Use recent interaction context to enrich the user query"""
    enriched_query = llm_api.generate(
        FR_USER_MULTITURN_QUESTION_SPECIFICATION.format(history=history, query=query),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return enriched_query


class RAGError(Exception):
    pass


class RAGBackEnd:
    def __init__(self, login, **kwargs):
        logger.debug("(Re)Initialization of chatbot backend")
        self._use_cache = kwargs.get("use_cache")
        self._llm_api = initialize_llm_api_client(kwargs.get("llm_api_name"))
        self._embedding_api = initialize_embedding_api_client()
        self._embedding_model_name = self._embedding_api.get_api_model_name()
        self._retrieval_mode = kwargs.get("retrieval_mode")
        self._reranker_model = (
            initialize_reranker_model(kwargs.get("reranker_model_name"))
            if self._retrieval_mode == RETRIEVAL_HYBRID_RERANKER
            else None
        )
        self._document_collection = self.load_document_collection(login)
        self._document_filter = None

    @property
    def document_collection(self):
        return self._document_collection

    @property
    def llm_api(self):
        return self._llm_api

    @property
    def embedding_api(self):
        return self._embedding_api

    @property
    def reranker_model(self):
        return self._reranker_model

    @property
    def bm25_model(self):
        return self._bm25_model


    def load_document_collection(self, login) -> DocumentCollection:
        """Retrieves the document collection the user can access to"""
        #TODO: improve in case of user has access to multiple collections
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
        #TODO: improve parameter config
        retriever = self.document_collection.collection.as_retriever(
            search_type = "mmr",
            search_kwargs={
                "k": 4, # number of retrieved docs
                "filter": {} if not self._document_filter else self._document_filter,
            })
        docs = retriever.get_relevant_documents(question)
        logger.debug(f'Number of retrieved docs = {len(docs)}')
        logger.debug(f"Relevant documents: {docs}")

        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(input_variables=['context', 'expected_answer_size', 'query'],
                                    messages=[HumanMessagePromptTemplate(
                                        prompt=PromptTemplate(input_variables=['context', 'expected_answer_size', 'query'],
                                                              template=FR_USER_BASE_RAG)
                                    )])

        # - LLM service
        #TODO - handle parameter: localLLM vs chatgpt
        llm = ChatOpenAI(openai_api_key = "EMPTY",
                         openai_api_base = "http://localhost:8888/v1",
                         model_name = "bofenghuang/vigostral-7b-chat"
        )
        #model_name="gpt-3.5-turbo", temperature=0)

        # - RAG chain
        rag_chain = (
                {"context": retriever | format_docs, "expected_answer_size": get_detail_level, "query": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        # TODO: implement reranking (optional)
        logger.debug(f"Calling RAG chain for question {question}...")
        answer = rag_chain.invoke(question)
        return answer

def get_detail_level(self):
    return "detailed"


    """
    def initialize_data(self, data_paths: List[Path]):
        Initializes data (list of paths, may be limited to a single element if
        only one file is loaded: load the data, recreates embedding if needed or use previously
        embeddings available in cache
        data_list = None
        embeddings_array = None
        for data_path in data_paths:
            print(data_path)
            data, embs = load_data(
                data_path,
                self.embedding_api,
                embedding_model_name=self._embedding_model_name,
                use_cache=self._use_cache,
            )
            data_list = (
                data
                if data_list is None
                else pd.concat([data_list, data], axis=0).reset_index(drop=True)
            )
            embeddings_array = (
                embs
                if embeddings_array is None
                else np.concatenate((embeddings_array, embs))
            )
        self.data = data_list
        self.embeddings = embeddings_array
        self._bm25_model = (
            make_docs_BM25_indexing(self.data)
            if self._retrieval_mode
            in (RETRIEVAL_BM25, RETRIEVAL_HYBRID, RETRIEVAL_HYBRID_RERANKER)
            else None
        )
    """


""""
    def query_oracle(self, query: str, history=None, **kwargs):
        if self.data is None:
            msg = "Data not provided!"
            logger.error(msg)
            raise RAGError(msg)
        if self.embeddings is None:
            msg = "Embeddings not computed!"
            logger.error(msg)
            raise RAGError(msg)

        enriched_query = query
        if kwargs.get("remember_recent_messages"):
            enriched_query = enrich_query(self.llm_api, query, history)
            logger.debug(enriched_query)
        else:
            history = ""

        (
            relevant_extracts,
            relevant_extracts_similarity,
        ) = extract_n_most_relevant_extracts(
            query=enriched_query,
            top_n=kwargs.get("top_n_extracts"),
            data=self.data,
            docs_embeddings=self.embeddings,
            embedding_api=self.embedding_api,
            bm25_model=self.bm25_model,
            retrieval_mode=kwargs.get("retrieval_mode"),
            reranker_model=self.reranker_model,
            similarity_threshold=kwargs.get("similarity_threshold"),
        )

        # Generates prompt
        prompt = generate_RAG_prompt(
            query,
            [extract[TEXT_COLUMN] for extract in relevant_extracts],
            expected_answer_size=kwargs.get("expected_answer_size"),
            custom_prompt=kwargs.get("custom_prompt"),
            history=history,
        )
        logger.debug(f"Prompt : {prompt}")
        # Generates response
        stream_response = self.llm_api.generate(
            prompt,
            # system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        return relevant_extracts, relevant_extracts_similarity, stream_response

    def simple_query(self, query: str, history=None, **kwargs):
        enriched_query = query
        if kwargs.get("remember_recent_messages"):
            enriched_query = enrich_query(self.llm_api, query, history)
            logger.debug(enriched_query)
        else:
            history = ""

        # Generates prompt
        prompt = generate_query_prompt(
            enriched_query,
            custom_prompt=kwargs.get("custom_prompt"),
            history=history,
        )
        logger.debug(f"Prompt : {prompt}")
        # Generates response
        stream_response = self.llm_api.generate(
            prompt,
            # system_prompt=FR_SYSTEM_DODER_RAG, -> NOT WORKING WITH CERTAIN MODELS (MISTRAL)
            temperature=TEMPERATURE,
            stream=True,
        )
        return stream_response
"""


