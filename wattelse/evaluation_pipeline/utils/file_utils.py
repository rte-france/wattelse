from pathlib import Path
from datetime import datetime
from loguru import logger


def handle_output_path(path: Path, overwrite: bool) -> Path:
    """Handle file path logic based on overwrite parameter."""
    if not path.exists() or overwrite:
        if path.exists() and overwrite:
            logger.info(f"Overwriting existing file: {path.name}")
        return path

    # If not overwriting, create a new filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    logger.warning(f"File already exists. Using alternative path: {new_path.name}")
    return new_path
