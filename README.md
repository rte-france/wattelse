# WattElse project

## Short description

WattElse is a NLP suite developed for the needs of RTE (Réseau de Transport d'Electricité).

It is composed of two main modules:
- a Retrieval Augmented Generation (RAG) application -> [wattelse/chatbot](wattelse/chatbot)
- a module to study topic evolution over time -> [wattelse/bertopic](wattelse/bertopic)

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

WattElse uses embedding models for *RAG* and *BERTopic*. It also uses larger generative models for *RAG* responses. By default, all models are loaded on GPU. For *RAG*, you will for example need:
- 1 GPU with > 20Go (or several smaller GPUs)


## Launching the NLP services

To start all services:
```bash
./start_all_services.sh
```

It will create a separate screen for each service running in the background.

To stop stop all services:
```bash
./stop_all_services.sh
```


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
