from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings


class VLLMConfig(BaseSettings):
    host: str
    port: str
    port_controller: str
    port_worker: str
    model_name: str
    cuda_visible_devices: str


# Load config file
config_file = Path(__file__).parent / "default_config.toml"
with open(config_file, "rb") as f:
    CONFIG = VLLMConfig(**tomllib.load(f))
