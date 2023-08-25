# Installation de Vigogne comme API

Buts 
- mettre à disposition de tous les services qui en ont besoin un accès au modèle de langage
- mutualiser les ressources

## Prérequis: installation de fastchat

``` pip install "fschat[model_worker,webui]```

## Lancement du service sur le serveur GPU

3 scripts à lancer

```python -m fastchat.serve.controller```


```CUDA_VISIBLE_DEVICES=0,2 python -m fastchat.serve.model_worker --model-path bofenghuang/vigogne-2-7b-instruct --gpus 1,2 --num-gpus 2```

```python -m fastchat.serve.openai_api_server --host localhost --port 8000```

## Scripts

Scripts are provided to start / stop the LLM service on the GPU server
- `start_llm_service.sh` - preferably run this command within a `screen` terminal
  (`screen -S llm` before launching the script and then `CTRL + A d` to detach the screen) 
- `stop_llm_service.sh`

## Test du service

```commandline
import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# First model
models = openai.Model.list()
model = models["data"][0]["id"]

# Chat completion API
chat_completion = openai.api_resources.Completion.create(
    model=model,
    prompt="Décris en un ou deux mots le thème associé à l'ensemble des mots clés suivants: hiver, électricité, risque, ecowatt",
    max_tokens=1024,
    temperature=0.7,
)
print("Chat completion results:", chat_completion)
```


