# WattElse project

## Short description

WattElse is a NLP suite developed for the needs of RTE (Réseau de Transport d'Electricité).

It is composed of two main modules:
- a Retrieval Augmented Generation (RAG) application -> **WattElse Doc**
- a simple chatbot interface to deploy and interact with any LLM -> **WattElse GPT**

Some services are used by several applications/users at the same time. To optimize resource use, these services are implemented in the form of APIs. A description of these services is available in [wattelse/api](wattelse/api).

WattElse also includes helper modules that provide additional functionalities such as summaries, web scrapping, and document parsing.

## Installation

Before trying to install WattElse, you first need to ensure you have:
- python >= 3.10
- sqlite3 >= 3.35

Then, create a virtual environnement:

```bash
python3 -m venv ~/.venv/wattelse-venv
source ~/.venv/wattelse-venv/bin/activate
```

You can then install the project dependencies with the following command:

```bash
./install.sh
```

## Hardware requirements

WattElse uses embedding models for *RAG*. It also uses larger generative models for responses. By default, all models are loaded on GPU. For *RAG*, you will for example need:
- 1 GPU with > 20Go (or several smaller GPUs)

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
        RAG[RAG Orchestrator]
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

## Sequence diagram for RAG
```mermaid


sequenceDiagram
    participant User
    participant LLM as LLM Service
    participant RAG as RAG Orchestrator
    participant Embedding as Embedding Service
    participant VectorDB as Vector Database
    participant DocStore as Document Store
    participant Parser as Document Parser
    participant Chunker as Text Chunker

    User->>RAG: Submit Query
    RAG->>Embedding: Generate Query Embedding
    Embedding-->>RAG: Query Vector

    RAG->>VectorDB: Semantic Search
    VectorDB-->>RAG: Relevant Document IDs

    RAG->>DocStore: Fetch Documents
    DocStore-->>RAG: Raw Documents

    RAG->>Parser: Parse Documents
    Parser-->>RAG: Structured Content

    RAG->>Chunker: Split Content
    Chunker-->>RAG: Text Chunks

    RAG->>LLM: Generate Response (Query + Context)
    LLM-->>RAG: Generated Response
    RAG-->>User: Final Answer

    Note over DocStore,Chunker: Document Processing Pipeline
    Note over RAG,VectorDB: Retrieval Pipeline
    Note over RAG,LLM: Generation Pipeline
```
