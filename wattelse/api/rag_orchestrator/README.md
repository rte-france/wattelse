# RAG Orchastrator API

This API is used in the RAG application. It provides a simple interface to handle multiple instances of `RAGBackend()`, see [rag_backend.py](../../chatbot/backend/rag_backend.py). Multiple instances of `RAGBackend()` are needed to cleanly separate the different groups of users in the RAG application: each group of users should have access to its collection of documents only.

## How to use

Ensure the API is launched.

```python
from pathlib import Path
from wattelse.api.rag_orchestrator.client import RAGOrchestratorClient

rag_api_endpoint = "https://localhost:1978"

# Initialize the API
api = RAGOrchestratorClient(url=rag_api_endpoint)

# Create a session for your group
api.create_session("TEST_SESSION")

# Check current sessions
api.get_current_sessions()

# Add a document to your session
my_file = Path("my_file.txt")
api.upload_files("TEST_SESSION", [my_file])

# List available documents
api.list_available_docs("TEST_SESSION")

# Ask a question about your collection
api.query_rag("TEST_SESSION", "De quoi parle le document ?")

# Clear your collection
# !!! WARNING: This will delete all documents in the session !!!
api.clear_collection("TEST_SESSION")
```
