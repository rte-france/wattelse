from pathlib import Path

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient


def test_rag_client():
    docs = [
        "NMT - Avenant n°4 de révision à l'accord sur le temps de travail.pdf",
        "NMT - Accord télétravail.pdf"
    ]

    files = [Path(__file__).parent.parent.parent / "data" / "chatbot" / "drh" / doc for doc in docs]

    # Test upload
    rag_client = RAGOrchestratorClient("bob", "bob")
    assert rag_client.session_id is not None

    available_docs = rag_client.list_available_docs()
    assert len(available_docs) == 0
    rag_client.upload_files(files)
    available_docs_after_upload = rag_client.list_available_docs()
    assert len(available_docs_after_upload) == len(available_docs) + 2

    # Test delete doc
    rag_client.remove_documents([docs[0]])
    available_docs_after_remove = rag_client.list_available_docs()
    assert len(available_docs_after_remove) == len(available_docs_after_upload) - 1

    # Test current sessions
    sessions = rag_client.get_current_sessions()
    assert rag_client.session_id in sessions

    # Test RAG query
    # query rag based on selected document collection
    anwser = rag_client.query_rag("Y'a une prime pour le télétravail?")
    assert anwser is not None

    # Clean all docs
    available_docs = rag_client.list_available_docs()
    rag_client.remove_documents(available_docs)
    assert len(rag_client.list_available_docs()) == 0
