# RAGBackend default configurations

This folder contains default configurations for `RAGBackend` class. Each configuration should be a `.toml` file. When instanciating a `RAGBackend`, the configuration is validated using the `RAGBackendConfig` pydantic model defined in [settings.py](settings.py).

[__init__.py](__init__.py) file contains the mapping between a config file and a config id to use when instanciating a `RAGBackend`.
