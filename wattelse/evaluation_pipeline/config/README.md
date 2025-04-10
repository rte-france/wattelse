# Configuration

This directory contains configuration management components for the RAG evaluation pipeline.

## Files

- `eval_config.py` - Configuration class for RAG evaluation settings
- `server_config.py` - Configuration class for server deployment settings
- `jury_01.toml` - Default configuration file for evaluation metrics and models
- `server_config.toml` - Default configuration file for server settings

## Configuration Management

The configuration system uses a combination of:
- INI-style config files (*.toml)
- Python dataclasses for type-safe configuration parsing
- Environment variables for deployment-specific settings

## Evaluation Configuration

The `EvalConfig` class handles:
- Loading and validating evaluation settings
- Managing model-specific configurations
- Providing access to prompts and regex patterns
- Tracking enabled/disabled metrics

Example section from `jury_01.toml`:

```ini
[EVAL_CONFIG]
enabled_metrics = ["faithfulness", "correctness", "retrievability"]

# for local-hosted models
[MODEL_META_LLAMA_META_LLAMA_3_8B_INSTRUCT]
model_name = "$LOCAL_OPENAI_DEFAULT_MODEL_NAME"
prompt_type = "meta-llama-3-8b"
regex_type = "re_llama3"
temperature = 0.0

# for cloud-hosted models
[MODEL_WATTELSE_GPT4O_MINI_SWEDEN_DEV]
model_name = "$AZURE_SE_WATTELSE_OPENAI_DEFAULT_MODEL_NAME_DEV"
deployment_type = "cloud"
api_base = "$AZURE_SE_WATTELSE_OPENAI_ENDPOINT_DEV"
api_key = "$AZURE_SE_WATTELSE_OPENAI_API_KEY_DEV"
prompt_type = "vanilla"
regex_type = "default"
temperature = 0.0

...
```

## Server Configuration

The `ServerConfig` class manages:
- Server host and port settings
- CUDA device allocation
- Controller and worker port assignments

Example section from `server_config.toml`:

```ini
[SERVER_CONFIG]
host = "0.0.0.0"
port = 8888
port_controller = 21001
port_worker = 21002
cuda_visible_devices = [0, 1]
```

## Usage

```python
from wattelse.evaluation_pipeline.config.eval_config import EvalConfig, ServerConfig

# Load configurations
eval_config = EvalConfig("path/to/jury_01.toml")
server_config = ServerConfig("path/to/server_config.toml")

# Access configuration values
enabled_metrics = eval_config.enabled_metrics
server_port = server_config.port
```