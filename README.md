# Wattelse project

## Short description

Wattelse is a NLP suite developed for the needs of RTE (Réseau de Transport d'Electricité).

It is composed of several modules:
- a RAG (Retrieval Augmented Generation) application
- a module to study topic evolution over time
- helper modules that provide additional functionalities such as summaries

## Requirements

### Hardware requirements

- 1 GPU with > 20Go (or several smaller GPUs) for RAG features

### Software requirements

- Python >= 3.10
- sqlite3 >= 3.35


## Installation

- create a Python virtual environment
- install the package in development mode (local installation) using the following command:

```:~/dev/wattelse/ $ pip install -e .```

## Launching the NLP services

### LLM service
The LLM service is a ChatGPT-like API that allows to run on local GPU(s) a LLM that can then be used by any application.
Two kind of APIs are available:

- an OpenAI compatible service 
  - see `wattelse/api/fastchat`
  - `fastchat_api.cfg` provides the information about which LLM to use (by default a French LLM) and on which port the 
  API is accessible. This file can be edited if you want to try another LLM or change the host and port.
  - the `start.sh` and `stop.sh` scripts allow to start/stop the service.
  - a Python client for this API is also provided in order to simplify the usage of the API

- a OLLAMA compatible service
  - see `wattelse/api/ollama`
  - `ollama_api.cfg` provides the information about which LLM to use (by default a French LLM) and on which port the 
  API is accessible. This file can be edited if you want to try another LLM or change the host and port.
  - the `start.sh` and `stop.sh` scripts allow to start/stop the service.
  - a Python client for this API is also provided in order to simplify the usage of the API

### Embedding service
The Embedding service returns a vectorized representation of the data provided as inputs.
- The encoder to be used by the embedding service is described in `wattelse/api/embedding/embedding_api.cfg`.
- The default encoder is better for French text, but can be changed according to your needs.
- a Python client for this API is also provided in order to simplify the usage of the API
- the `start.sh` and `stop.sh` scripts allow to start/stop the service.


### RAG service
The RAG service is able to manage separate users with different spaces for different document collections.
- the `start.sh` and `stop.sh` scripts allow to start/stop the service.

### RAG application

The RAG application is a Django application that uses all the services listed above: LLM, Embedding, RAG.

You have to ensure that all these services are running prior to run the RAG application.
- the code is available in: `wattelse/chatbot/frontend`.
- the `start.sh` scripts allow to start the application
- by default, the RAG application runs at this URL: http://localhost:8000

**Important information**

The LLM, Embedding services are using GPU(s). Adapt the `start.sh` scripts according to your configuration.

## Topic demonstrator

- Code available in: `wattelse/bertopic`
- To run the demonstrator:

  - `cd wattelse/bertopic`
  - streamlit run app/Main_pages.py
