#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pathlib import Path

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

TEST_GROUP_NAME = "pytest_group"


def test_rag_client():
    docs = [
        "NMT - Avenant n°4 de révision à l'accord sur le temps de travail.pdf",
        "NMT - Accord télétravail.pdf",
    ]

    files = [
        Path(__file__).parent.parent.parent / "data" / "chatbot" / "drh" / doc
        for doc in docs
    ]

    # Set up RAG client
    rag_client = RAGOrchestratorClient()
    rag_client.clear_collection(TEST_GROUP_NAME)
    rag_client.create_session(TEST_GROUP_NAME)

    # Test current sessions
    sessions = rag_client.get_current_sessions()
    assert TEST_GROUP_NAME in sessions

    # Test upload
    available_docs = rag_client.list_available_docs(TEST_GROUP_NAME)
    assert len(available_docs) == 0
    rag_client.upload_files(TEST_GROUP_NAME, files)
    available_docs_after_upload = rag_client.list_available_docs(TEST_GROUP_NAME)
    assert len(available_docs_after_upload) == len(available_docs) + 2

    # Test delete doc
    rag_client.remove_documents(TEST_GROUP_NAME, [docs[0]])
    available_docs_after_remove = rag_client.list_available_docs(TEST_GROUP_NAME)
    assert len(available_docs_after_remove) == len(available_docs_after_upload) - 1

    # Test RAG query
    # query rag based on selected document collection
    anwser = rag_client.query_rag(
        TEST_GROUP_NAME, "Y'a une prime pour le télétravail ?"
    )
    assert anwser is not None

    # Clean all docs
    rag_client.clear_collection(TEST_GROUP_NAME)
    assert TEST_GROUP_NAME not in rag_client.get_current_sessions()
