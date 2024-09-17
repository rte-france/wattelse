# APIs

Some services are used by several applications/users at the same time. To optimize resource use, these services are implemented in the form of APIs.

## APIs description

Available APIs:

- [embedding](embedding): uses a SentenceTransformer model to transform text into embeddings
- [vllm](vllm): uses a LLM for text generation by starting an OpenAI API like endpoint
- [rag_orchestrator](rag_orchastrator): used for multi-user RAG, redirects queries to the appropriate RAG instance
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


## Fonctionnement commun aux API

Pour lancer une API :
```source wattelse/api/{nom_api}/start.sh```

Pour arrêter une API :
```source wattelse/api/{nom_api}/stop.sh```
(nécessite les droits sudo pour arrêter les API lancées par d'autres utilisateurs)




### EmbeddingAPI

Une fois l'API lancée :
```
from wattelse.api.embedding.class_embedding_api import EmbeddingAPI
api = EmbeddingAPI()
text = ['text1', 'text2']
embeddings = api.encode(text)
```

### FastchatAPI

Une fois l'API lancée :
```
from wattelse.api.fastchat.class_fastchat_api import FastchatAPI
api = FastchatAPI()
prompt = "C'est quoi RTE ? Répond en 1 phrase courte."
answer = api.generate(prompt)
```

### OllamaAPI

Une fois l'API lancée :
```
from wattelse.api.ollama.class_ollama_api import OllamaAPI
api = OllamaAPI()
prompt = "C'est quoi RTE ? Répond en 1 phrase courte."
answer = api.generate(prompt)
```

### OpenAI_API

Une fois l'API lancée :
```
from wattelse.api.openai.class_openai_api import OpenAI_API
api = Open_AIAPI()
prompt = "C'est quoi RTE ? Répond en 1 phrase courte."
answer = api.generate(prompt)
```



