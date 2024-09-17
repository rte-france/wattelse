# RAGBackend

The `RAGBackend` class allows to manage the RAG pipeline through a single class, including document parsing and chunking, embedding, information retrieval and answer generation.

## How to use

Before using the `RAGBackend` class, ensure embedding and LLM API are launched.

```python
from pathlib import Path
from wattelse.chatbot.backend.rag_backend import RAGBackEnd

# Initialize the RAGBackend
rag = RAGBackEnd("test_backend")

# Add a document
path_to_doc = Path("test_document.pdf")
with open(path_to_doc, "rb") as f:
    rag.add_file_to_collection(path_to_doc.name, f)

# List available docs
print(rag.get_available_docs())

# Ask a question about docs
response = rag.query_rag("question ?")

# Print answer
print(response["answer"])

# Print relevant extracts
for extract in response["relevant_extracts"]:
    print(extract["content"])

# Clear the backend
# !!! WARNING: This will delete all documents and associated embeddings !!!
rag.clear_collection()
```