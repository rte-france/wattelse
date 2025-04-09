# WattElse project

WattElse is a NLP suite developed for the needs of RTE (Réseau de Transport d'Electricité).

It is composed of two main modules:

- a simple chatbot interface to interact with any LLM -> **WattElse GPT**
- a Retrieval Augmented Generation (RAG) application -> **WattElse DOC**

Some services are used by several applications/users at the same time. To optimize resource use, these services are implemented in the form of APIs. A description of these services is available in [wattelse/api](wattelse/api).

WattElse also includes helper modules that provide additional functionalities such as summaries, web scrapping, and document parsing.

## Installation

Before trying to install WattElse, you first need to ensure you have:

- `python >= 3.10`
- `sqlite3 >= 3.35`

Clone this project:

```bash
git clone https://github.com/rte-france/wattelse.git
```

Create a virtual environment:

```bash
cd wattelse
python -m venv .venv
source .venv/bin/activate
```

Install WattElse and its dependencies:

```bash
./install.sh
```

## Environnement variables

To run WattElse, you need to set the following environment variables:

- `WATTELSE_BASE_DIR`: path where WattElse data will be stored
- `DJANGO_SECRET_KEY`: Django secret key (see [Django documentation](https://docs.djangoproject.com/en/4.2/ref/settings/#std-setting-SECRET_KEY))

To create a Django `SECRET_KEY`, run the following code in a python shell:

```python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

## Django initialization

To initialize the Django database, follow these steps:

- Go to the django `web_app` folder:

```bash
cd wattelse/web_app
```

- Create Django tables:

```bash
python manage.py makemigrations
python manage.py migrate
```

- Create Django superuser:

```bash
python manage.py createsuperuser
```

- Start Django server:

```bash
python start.py
```

Django web app should be running at: http://localhost:8000

## Launch WattElse

To launch WattElse with all services, go to WattElse root folder and run:

```bash
./start_all_services.sh
```

This script starts all services in separated `screens`:

- Embedding API
- RAGOrchestrator API
- Django
-

## Hardware requirements

By default, WattElse only loads an embedding model on start. It requires around 2GB of VRAM if loaded on GPU.

The LLM used depends on the RAG config. By default, no local LLM is loaded so you need to link RAG config to a remote LLM (OpenAI, Azure...). For RAG config management, see [wattelse/rag_backend](wattelse/rag_backend).

If you want to load a local LLM using `vLLM`, you need to have enough VRAM to load the model. For example, the `llama-3.1-8B-instruct` model requires around 16GB of VRAM.

# RAG service

## Overview of main steps

```mermaid
flowchart TB
    subgraph Preprocessing["Data Preprocessing"]
        direction TB
        RawDocs["Raw Documents"]
        TextChunks["Text Chunks"]
        VectorDB[(Vector Database)]

        RawDocs --> |"Text Splitting"| TextChunks
        TextChunks --> |"Embedding Generation"| VectorDB
    end

    subgraph QueryProcessing["Query Processing"]
        direction TB
        UserQuery["User Query"]
        QueryEmbed["Query Embedding"]
        SimilaritySearch["Similarity Search"]
        ContextRetrieval["Context Retrieval"]

        UserQuery --> QueryEmbed
        QueryEmbed --> SimilaritySearch
        SimilaritySearch --> ContextRetrieval
    end

    subgraph ResponseGeneration["Response Generation"]
        direction TB
        LLM["Large Language Model"]
        Response["Generated Response"]

        ContextRetrieval --> LLM
        UserQuery --> LLM
        LLM --> Response
    end

    VectorDB --> SimilaritySearch

```

## Description of components

```mermaid
flowchart TB
    subgraph Input
        User[User Interface]
    end

    subgraph "RAG Core Components"
        RAG[RAG Backend]
        LLM[LLM Service]
        Embedding[Embedding Service]
    end

    subgraph "Storage Layer"
        VectorDB[(Vector Database)]
        DocStore[(Document Store)]
    end

    subgraph "Document Processing"
        Parser[Document Parser]
        Chunker[Text Chunker]
    end

    User <--> RAG
    RAG <--> LLM
    RAG <--> Embedding
    RAG <--> VectorDB
    RAG <--> DocStore
    DocStore <--> Parser
    Parser <--> Chunker

    classDef core fill:#e1f5fe,stroke:#01579b
    classDef storage fill:#e8f5e9,stroke:#1b5e20
    classDef processing fill:#fff3e0,stroke:#e65100
    classDef input fill:#f3e5f5,stroke:#4a148c

    class RAG,LLM,Embedding core
    class VectorDB,DocStore storage
    class Parser,Chunker processing
    class User input

```

## Simplified sequence diagram for RAG

```mermaid


sequenceDiagram
    participant User
    participant UI as User Frontend
    participant LLM as LLM Service
    participant RAG as RAG Backend
    participant Embedding as Embedding Service
    participant VectorDB as Vector Database
    participant DocStore as Document Store
    participant Parser as Document Parser
    participant Chunker as Text Chunker

    User->>UI: Authenticate
    UI-->>User: Authentication OK

    UI->>UI: Create Session

    User->>UI: Upload files
    UI->>RAG: New files
    RAG->>VectorDB: Check cache
    VectorDB->>VectorDB: check file names
    VectorDB-->>RAG: Files not in cache
    RAG->>Parser: Parse files(files)
    Parser->>Chunker: Chunk(documents)
    Chunker-->>RAG: documents chunks
    RAG->>DocStore: store(files)
    RAG->>VectorDB: store(documents chunks)
    VectorDB->>Embedding: compute embeddings(documents chunks)
    Embedding->>Embedding: compute
    Embedding-->>VectorDB: computed embeddings
    VectorDB->>VectorDB: store
    VectorDB-->>RAG: storage ok

    User->>UI: <br><br><br><br>Enter Query
    UI->>RAG: Submit Query
    RAG->>Embedding: Generate Query Embedding
    Embedding-->>RAG: Query Vector

    RAG->>VectorDB: Semantic Search
    VectorDB->>VectorDB: Similarity function
    VectorDB-->>RAG: Relevant Document Extracts

    RAG->>LLM: Generate Response (Query + Context + (history))
    LLM-->>RAG: Generated Response
    RAG-->>UI: Final Answer
    UI-->>User: Displayed answer

    User-->>UI: <br>Provide feedback
    UI-->>UI: Store feedback



    User->>UI: <br><br><br>Logout / Timeout
    UI->>UI: Kill Session

    Note over DocStore,Chunker: Document Processing Pipeline
    Note over RAG,VectorDB: Retrieval Pipeline
    Note over RAG,LLM: Generation Pipeline
```

## Main code dependencies

```mermaid
flowchart TB
   wattelse["wattelse"]

   subgraph ML["Machine Learning"]
       torch["torch"]
       sklearn["scikit-learn"]
       accelerate["accelerate"]
       scipy["scipy"]
       numpy["numpy"]
   end

   subgraph LLM["LLM & RAG"]
       langchain["langchain"]
       langchain_comm["langchain-community"]
       langchain_chroma["langchain-chroma"]
       langchain_openai["langchain-openai"]
       llama_index["llama-index-core"]
       openai["openai"]
       chromadb["chromadb"]
       sent_trans["sentence-transformers"]
       vllm["vllm"]
       fschat["fschat"]
       tiktoken["tiktoken"]
   end

   subgraph Web["Web Framework"]
       django["django"]
       fastapi["fastapi"]
       streamlit["streamlit"]
       uvicorn["uvicorn"]
   end

   subgraph Doc["Document Processing"]
       docxtpl["docxtpl"]
       python_docx["python-docx"]
       python_pptx["python-pptx"]
       pymupdf["pymupdf"]
       unstructured["unstructured"]
       mammoth["mammoth"]
       xlsx2html["xlsx2html"]
   end

   subgraph Data["Data Processing"]
       pandas["pandas"]
       plotly["plotly"]
       seaborn["seaborn"]
       bs4["bs4"]
   end

   wattelse --> ML
   wattelse --> LLM
   wattelse --> Web
   wattelse --> Doc
   wattelse --> Data
```
