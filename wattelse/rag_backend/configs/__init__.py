from pathlib import Path

# Mapping between the configuration name as it should appear in Django (file name without extension and full path
CONFIG_NAME_TO_CONFIG_PATH = {
    path.stem: path for path in Path(__file__).parent.glob("*.toml")
}
