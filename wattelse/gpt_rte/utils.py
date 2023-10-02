import socket
from pathlib import Path

from wattelse.common.vars import GPU_SERVERS

MODEL_BASE_PATH = (
    Path("/data/weak_signals/models/")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "models"
)
