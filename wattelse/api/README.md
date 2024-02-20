# Installation de différents services comme API

Buts :
- mettre à disposition de tous les services qui en ont besoin un accès au modèle de langage
- éviter de charger plusieurs fois le même modèle pour des applications différentes

API disponibles :
- embedding : modèle SentenceTransformer pour transformer un texte en embedding
- fastchat : LLM pour générer du texte à partir d'un prompt
- ollama : identique à fastchat mais permet de quantifier les modèles (utile notamment pour utiliser Mixtral 8*7B)
- openai : identique a fastchat mais permet de requêter l'api d'openai

## Fonctionnement commun aux API

Pour lancer une API :
```source wattelse/api/{nom_api}/start.sh```

Pour arrêter une API :
```source wattelse/api/{nom_api}/stop.sh```
(nécessite les droits sudo pour arrêter les API lancées par d'autres utilisateurs)

Tous les paramètres permettant de gérer l'API sont dans le fichier de configuration :
```wattelse/api/{nom_api}/{nom_api}_api.cfg```

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



