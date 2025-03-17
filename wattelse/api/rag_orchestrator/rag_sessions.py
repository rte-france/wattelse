from wattelse.chatbot.backend.rag_backend import RAGBackend

# Global session storage
RAG_SESSIONS: dict[str, RAGBackend] = {}
