<<<<<<< HEAD
# RAG

This folder is an implementation of a RAG (Retrieval-Augmented Generation) system.

## Folders description

- [backend](backend): implementation of the `RAGBackend` class. It allows to manage the RAG pipeline through a single class, including document parsing and chunking, embedding, information retrieval and answer generation.
- [frontend](frontend): RAG application front interface using [Django](https://www.djangoproject.com/) and dashboard to monitor the usage.
- [eval](eval): Evaluation scripts for the RAG system.

## How to launch RAG system

Ensure you have WattElse installed in your python environment. Then go to the root of this repository and launch the following script:

```bash
./start_all_services.sh
```

This will launch all services for the RAG system to work in separated `screens`: embedding, LLM, RAG Orchestrator, Django frontend and dashboard.

To stop the RAG system:

```bash
./stop_all_services.sh
```

**Warning: All models are loaded on GPU by default (LLM and embedding), ensure you have enought GPU memory for the models you want to serve.**
=======
# Django web app

This folder contains the Django web app for WattElse. It is divided into several subfolders:

- [accounts](accounts): user authentication and management(login, logout, password management)
- [home](home): home page where user can choose between the different apps
- [gpt](gpt): GPT app for chatting with an LLM assistant
- [doc](doc): DOC app for the RAG system
>>>>>>> e3a7b4b (First working version of django refactor)
