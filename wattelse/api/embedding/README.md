# How to use

Ensure the API is launched.

```python
from wattelse.api.embedding.client import EmbeddingAPIClient

embedding_service_endpoint = "https://yourservice:1234"

# Initialize the API
api = EmbeddingAPIClient(embedding_service_endpoint)

# Get embedding model name
print(api.get_api_model_name())

# Transform text to embedding
text = "Hello, world!"
embedding = api.embed_query(text)
print(embedding)
```