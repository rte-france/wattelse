from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    model_name: str
    port: int
    number_workers: int
    cuda_visible_devices: str


# Load config file
config_file = Path(__file__).parent / "config.toml"
with open(config_file, "rb") as f:
    CONFIG = Config(**tomllib.load(f))
