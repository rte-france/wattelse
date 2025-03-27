from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings


class DjangoConfig(BaseSettings):
    host: str
    port: str


# Load config file
config_file = Path(__file__).parent / "default_config.toml"
with open(config_file, "rb") as f:
    CONFIG = DjangoConfig(**tomllib.load(f))
