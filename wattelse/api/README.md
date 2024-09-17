# APIs

Some services are used by several applications/users at the same time. To optimize resource use, these services are implemented in the form of APIs.

## APIs description

Available APIs:

- [embedding](embedding): uses a SentenceTransformer model to transform text into embeddings
- [vllm](vllm): uses a LLM for text generation by starting an OpenAI API like endpoint
- [rag_orchestrator](rag_orchestrator): used for multi-user RAG, redirects queries to the appropriate RAG instance
- [openai](openai): provides a simple interface to interact with the OpenAI API
- [fastchat](fastchat): deprecated
- [ollama](ollama): deprecated

## How to launch

To launch an API, go to the specific API folder and run the `start.sh` script. For example:

```bash
cd wattelse/api/embedding
./start.sh
```

To stop an API, run the `stop.sh` script:

```bash
./stop.sh
```

All parameters used to manage the API are in the config file: `wattelse/api/{nom_api}/{nom_api}_api.cfg`