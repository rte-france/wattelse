# APIs

Some services are used by several applications/users at the same time. To optimize resource use, these services are implemented in the form of APIs.

## APIs description

Available APIs:

- [embedding](embedding): uses a SentenceTransformer model to transform text into embeddings
- [vllm](vllm): uses a LLM for text generation by starting an OpenAI API like endpoint
- [rag_orchestrator](rag_orchestrator): used for multi-user RAG, redirects queries to the appropriate RAG instance
- [openai](openai): client only provides a simple interface to interact with the OpenAI API
- [fastchat](fastchat): deprecated
- [ollama](ollama): deprecated

## Start an API

To launch an API, go to the specific API folder and run the `start.py` script. For example:

```bash
cd wattelse/api/embedding
python start.py
```

The API will be launched with the default configuration file `config/default_config.toml`.

It is possible to launch multiple `embedding` or `vllm` APIs in parallel with different configurations. To do so, create a new configuration file in the `config` folder and run the `start.py` script with the correct environment variable set:

- `EMBEDDING_API_CONFIG_FILE`: path to the configuration file to use for the `embedding` API
- `VLLM_API_CONFIG_FILE`: path to the configuration file to use for the `vllm` API

For example:

```bash
# Terminal 1: embedding API with default config
cd wattelse/api/embedding
python start.py

# Terminal 2: embedding API with custom config
cd wattelse/api/embedding
EMBEDDING_API_CONFIG_FILE=config/custom_config.toml python start.py
```


## Stop an API

If the API is running in your current terminal, you can stop it by pressing `Ctrl+C`.

To stop an API that is running in the background, run the `stop.py` script:

```bash
python stop.py
```