#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Union, BinaryIO
import uuid

from langchain_core.documents import Document
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from loguru import logger

from wattelse.chatbot.backend import DATA_DIR
from wattelse.chatbot.backend.vector_database import format_docs, \
    load_document_collection

from wattelse.chatbot.backend.prompts import EN_USER_MULTITURN_QUESTION_SPECIFICATION, CORRECT_ANSWER_SPECIFICATION, \
    EXTRACT_UPDATE_PROMPT, EN_USER_RAG_NEW, EN_SYSTEM_PROMPT_RAG, EN_SYSTEM_QUERY_CONTEXTUALIZATION, \
    EN_USER_QUERY_CONTEXTUALIZATION, EN_USER_BASE_RAG
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
        llm_config = {"openai_api_key": os.getenv('OPENAI_API_KEY', None),
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


def preprocess_streaming_data(streaming_data):
    # TO DO: Modify this according to my update functions needs
    """Generator to preprocess the streaming data coming from LangChain `rag_chain.stream()`.
    First sent chunk contains relevant_extracts in a convenient format.
    Following chunks contain the actual response from the model token by token.
    """
    for chunk in streaming_data:
        context_chunk = chunk.get("context", None)
        if context_chunk is not None:
            relevant_extracts = [{"content": s.page_content, "metadata": s.metadata} for s in context_chunk]
            relevant_extracts = {"relevant_extracts": relevant_extracts}
            yield json.dumps(relevant_extracts)
        answer_chunk = chunk.get("answer")
        if answer_chunk:
            yield json.dumps(chunk)


def filter_history(history, window_size):
    # window size = question + answser, we return the last ones
    return history[-2 * window_size:]


class RAGBackEnd:
    def __init__(self, group_id: str):
        logger.info(f"[Group: {group_id}] Initialization of chatbot backend")
        # Load document collection
        self.document_collection = load_document_collection(group_id)
        self.document_filter = None

        # Retriever parameters
        self.top_n_extracts = retriever_config["top_n_extracts"]
        self.retrieval_method = retriever_config["retrieval_method"]
        self.similarity_threshold = retriever_config["similarity_threshold"]
        self.multi_query_mode = retriever_config["multi_query_mode"]

        # Generator parameters
        LLM_API_NAME = os.getenv('LLM_API_NAME', None) # check if environment variable specified while running container
        self.llm_api_name = LLM_API_NAME if LLM_API_NAME else generator_config["llm_api_name"]
        self.expected_answer_size = generator_config["expected_answer_size"]
        self.remember_recent_messages = generator_config["remember_recent_messages"]
        self.temperature = generator_config["temperature"]
        
        # Prompts
        self.system_prompt = EN_SYSTEM_PROMPT_RAG
        #self.user_prompt = EN_USER_RAG_NEW
        self.custom_prompt = EN_USER_BASE_RAG
        self.system_prompt_query_contextualization = EN_SYSTEM_QUERY_CONTEXTUALIZATION
        self.user_prompt_query_contextualization = EN_USER_QUERY_CONTEXTUALIZATION


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
        logger.debug(f"File {file_name} stored in: {path}")

        # Parse file
        logger.debug(f"Parsing: {path}")
        docs = parse_file(path)
        
        # Split the file into smaller chunks as a list of Document
        logger.debug(f"Chunking: {path}")
        splits = split_file(path.suffix, docs)
        logger.info(f"Number of chunks for file {file_name}: {len(splits)}")

        
        #add a unique id to each document's metadata to be able to update the document later
        for split in splits:
            unique_id = str(uuid.uuid4())  # Convert UUID to string
             # Add the unique identifier to the document's metadata
            split.metadata["unique_id"] = unique_id
   
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
    
    def get_doc_list(self, document_filter: Dict | None) -> Dict:
        """Returns the list of documents in the collection, using the current document filter"""
        data = self.document_collection.collection.get(include=["documents", "metadatas"],
                                                       where={} if not document_filter else document_filter)
        langchain_documents = []
        for doc, meta in zip(data["documents"], data["metadatas"]):
            langchain_doc = Document(page_content=doc, metadata=meta)
            langchain_documents.append(langchain_doc)
        return langchain_documents

    def get_text_list(self, document_filter: Dict | None) -> List[str]:
        """Returns the list of texts in the collection, using the current document filter"""
        data = self.document_collection.collection.get(include=["documents", "metadatas"],
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
        
    def get_document_by_uuid(self, document_ids: List[str]):
        """Create a filter on the document collection based on a list of file names"""
        if not document_ids:
            return None
        elif len(document_ids) == 1:
            return {"unique_id": document_ids[0]}
        else:
            return {"$or": [{"unique_id": f} for f in document_ids]}


    def select_by_keywords(self, keywords: List[str]):
        """Create a filter on the document collection based on a list of keywords"""
        # TODO: to be implemented
        self.document_filter = None
        
    
    
        

    def query_rag(self, question: str, history: List[dict[str, str]] = None,
                  selected_files: List[str] = None, stream: bool = False) -> Union[Dict, StreamingResponse]:
        """Query the RAG"""
        self.question = question #just so that i can test the feedback function
        # Sanity check
        if self.document_collection is None:
            raise RAGError("No active document collection!")
        
         # Get document filter
        self.document_filter = self.get_document_filter(selected_files)

        # Configure retriever
        search_kwargs = {
            "k": self.top_n_extracts,  # number of retrieved docs
            "filter": {} if not self.document_filter else self.document_filter,
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
            bm25_retriever = BM25Retriever.from_documents(self.get_doc_list(document_filter))
            bm25_retriever.k = self.top_n_extracts
            if self.retrieval_method == BM25:
                retriever = bm25_retriever
            else:  # ENSEMBLE
                dense_retriever = self.document_collection.collection.as_retriever(
                    search_type=SIMILARITY,
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
        
        """
        prompt = ChatPromptTemplate(input_variables=["context", "history", "query"],
                                    messages=[
                                        SystemMessagePromptTemplate(
                                            prompt=PromptTemplate(
                                                input_variables=[],
                                                template=self.system_prompt)
                                        ),
                                        HumanMessagePromptTemplate(
                                            prompt=PromptTemplate(
                                                input_variables=['context', "history",'query'],
                                                template=self.user_prompt)
                                    )])
"""
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
             "history": (lambda _: get_history_as_text(history)),
             "query": (lambda _: question),
             "expected_answer_size": self.get_detail_level, 
             "contextualized_query": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # TODO: implement reranking (optional)
        logger.debug(f"Calling RAG chain for question {question}...")

        # Handle conversation history
        contextualized_question = "query : " + self.contextualize_question(question, history)

        # Handle answer
    
        if stream:
            # stream response on server side...
            return preprocess_streaming_data(rag_chain.stream(contextualized_question))
            
        else:
            resp = rag_chain.invoke(contextualized_question)
            answer = resp.get("answer")
            sources = resp.get("context")
            # Transform sources
            relevant_extracts = [{"content": s.page_content, "metadata": s.metadata} for s in sources]
        
        # answer and retriever to be used also for user feedback and update if necessary
        self.answer=answer
        self.relevant_extracts = relevant_extracts

        # Return answer and sources
        return {"answer": answer, "relevant_extracts": relevant_extracts}

    def contextualize_question(self, message: str, history: List[dict[str, str]] = None,
                               interaction_window: int = 3) -> str:
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
            prompt = ChatPromptTemplate(input_variables=["history", "query"],
                                        messages=[
                                            SystemMessagePromptTemplate(
                                            prompt=PromptTemplate(
                                                input_variables=[],
                                                template=self.system_prompt_query_contextualization)
                                        ),
                                            HumanMessagePromptTemplate(
                                                prompt=PromptTemplate(
                                                    input_variables=["history", "query"],
                                                    template=self.user_prompt_query_contextualization)
                                        )])

            chain = (prompt
                     | self.llm
                     | StrOutputParser())

            # Format messages into a single string
            history_as_text = get_history_as_text(history)
            contextualized_question = chain.invoke({"query": message, "history": history_as_text})
            logger.debug(f"Contextualized question: {contextualized_question}")
            return contextualized_question


    def get_detail_level(self, question: str):
        """Returns the level of detail we wish in the answer. Values are in this range: {"courte", "détaillée"}"""
        return "short" if self.expected_answer_size == "short" else "détaillée"
    

    
    
    def handle_user_feedback(self):
        feedback = input("Do you think the answer was correct ? (yes/no)")
        if feedback == "yes":
            print("Thank you for your feedback")
        else:
            self.wrong_answer = self.answer
            #self.relevant_doc_ids = [doc.metadata["document_id"] for doc in self.relevant_extracts]
            self.uuids = [doc.get("metadata")["unique_id"] for doc in self.relevant_extracts]
            
            print("Please provide the correct answer")
            self.correct_answer = input()
            print("Thank you for your feedback")
        
        """correct_answer_template = CORRECT_ANSWER_SPECIFICATION
        new_extract = correct_answer_template.format(query=self.question, correct_answer=self.correct_answer)
        
        # add the correct answer to the document collection
        self.document_collection.add_texts(
            [new_extract], 
            #[correct_answer],
            metadata = [{"source": "corrected_extract", "file_name": ""}])
        """
        # consider adding an extract removr here in the case that the incorrect extract 
        # has the corrected_extracts metadata
    
    
    def update_extract_with_wrong_info(self):
        wrong_answer = self.wrong_answer
        
        #Get document filter, only the extracts that were used in the wrong answer
        #document_filter = self.get_document_by_id(self.relevant_doc_ids)
        document_filter = self.get_document_by_uuid(self.uuids)

        # Configure retriever
        search_kwargs = {
            "k": 1,  # number of retrieved docs
            #"filter": {"ids": "01139c96-c1d7-43ca-a7ca-c4946e096402"}
           "filter": {} if not document_filter else document_filter,
        }
      
        retrieval_method = "similarity"
        retriever = self.document_collection.collection.as_retriever(
                search_type=retrieval_method,
                search_kwargs=search_kwargs
            )
       
        
        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(input_variables=['document', 'correction'],
                                    messages=[HumanMessagePromptTemplate(
                                        prompt=PromptTemplate(
                                            input_variables=['document', 'correction'],
                                            template=EXTRACT_UPDATE_PROMPT)
                                    )])

        # - RAG chain
        rag_chain_from_docs = (
                RunnablePassthrough.assign(document=(lambda x: format_docs(x["document"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )
        # returns both answer and sources
        rag_chain = RunnableParallel(
            {"document": retriever,
             "correction": self.get_right_answer}
        ).assign(modified_extract=rag_chain_from_docs)
        
        
        # Handle Update
        resp = rag_chain.invoke(wrong_answer)
        modified_extract = resp.get("modified_extract")
        print("The document to update is: ",  resp.get("document"), "\n\n")
        print("The uuids of relevant extracts are: ", self.uuids, "\n\n")
        source_to_update = resp.get("document")[0]
        update_id = source_to_update.metadata["document_id"]
        metadata= self.document_collection.collection.get(update_id).get("metadatas")[0]
        new_document = Document(
            page_content=modified_extract,
            metadata=metadata)
        print("The previous extract was: \n", source_to_update.page_content, "\n\n")
        print("The updated extract is: \n", new_document.page_content)
        print("The id of the previous extract is: ", update_id, "\n\n")
              
        self.document_collection.collection.update_document(update_id, new_document)
        print("The extract has been updated")
        
    # callable to pass into RAG chain    
    def get_right_answer(self, wrong_answer : str):
        #Returns the right answer provided by user 
        return self.correct_answer
    

    
    def update_extract(self, relevant_extract_uuids: str, wrong_answer: str, correction: str):
        """Update extract using user feedback"""
        
        document_filter = self.get_document_by_uuid(relevant_extract_uuids)

        # Configure retriever
        search_kwargs = {
            "k": 1,  # number of retrieved docs
            #"filter": {"ids": "01139c96-c1d7-43ca-a7ca-c4946e096402"}
           "filter": {} if not document_filter else document_filter,
        }
      
        retrieval_method = "similarity"
        retriever = self.document_collection.collection.as_retriever(
                search_type=retrieval_method,
                search_kwargs=search_kwargs
            )
       
        
        # Definition of RAG chain
        # - prompt
        prompt = ChatPromptTemplate(input_variables=['document', 'correction'],
                                    messages=[HumanMessagePromptTemplate(
                                        prompt=PromptTemplate(
                                            input_variables=['document', 'correction'],
                                            template=EXTRACT_UPDATE_PROMPT)
                                    )])

        # - RAG chain
        rag_chain_from_docs = (
                RunnablePassthrough.assign(document=(lambda x: format_docs(x["document"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )
        # returns both answer and sources
        rag_chain = RunnableParallel(
            {"document": retriever,
             "correction": (lambda _: correction)}
        ).assign(modified_extract=rag_chain_from_docs)
        
        
        # Handle Update
        resp = rag_chain.invoke(wrong_answer)
        modified_extract = resp.get("modified_extract")
        
        
        source_to_update = resp.get("document")[0]
        update_id = source_to_update.metadata["document_id"]
        metadata= self.document_collection.collection.get(update_id).get("metadatas")[0]
        new_document = Document(
            page_content=modified_extract,
            metadata=metadata)
        
        
              
        self.document_collection.collection.update_document(update_id, new_document)
        logger.info("The extract has been updated")
        return {"previous_extract": source_to_update.page_content, 
                "updated_extract": new_document.page_content,
                "extract_id": update_id,
                "document": source_to_update.metadata["file_name"]}
        
        
        



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