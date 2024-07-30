# Custom OpenAI client quickstart

The `OpenAI_Client` class is based on the [OpenAI API](https://github.com/openai/openai-python) and provides a simple interface to interact with it. It enables calling LLM from different sources:
- Official OpenAI API, using OpenAI models (**be careful with confidential data !**)
- Azure OpenAI API, using LLMs deployed in Azure Cloud
- Local OpenAI compatible server, using locally deployed LLM (with [vLLM](https://docs.vllm.ai/en/latest/index.html) for example)

Before using it, you should set different environment variables:
- **(mandatory)** `OPENAI_API_KEY`: OpenAI API key for official or Azure OpenAI API. Set it to `EMPTY` for local LLM.
- **(optionnal)** `OPENAI_ENDPOINT`: if not set, `OpenAI_Client` will use official OpenAI API. Set it to `https://wattelse-openai.openai.azure.com/` to use Azure Cloud LLM. Set it to `http://localhost:8888/v1` (or any other local endpoint listening to an OpenAI compatible server) to use local LLM.
- **(optionnal)** `OPENAI_DEFAULT_MODEL_NAME`: specify what model to use by default.

Here is a quick example of how to use it:

```python
from wattelse.api.openai.client_openai_api import OpenAI_Client
api = OpenAI_Client()
api.generate("C'est quoi RTE ?")
```

See [`OpenAI_Client`](https://github.com/rte-france/wattelse/blob/main/wattelse/api/openai/client_openai_api.py) for more advanced usage.


To check what model is being used:

```python
print(api.model_name)
```