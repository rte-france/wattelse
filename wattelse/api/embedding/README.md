# How to use

Ensure the API is launched.

```python
from wattelse.api.embedding.client import EmbeddingAPI

# Initialize the API
api = EmbeddingAPI()

# Get embedding model name
print(api.get_api_model_name())

# Transform text to embedding
text = "Hello, world!"
embedding = api.embed_query(text)
print(embedding)
```