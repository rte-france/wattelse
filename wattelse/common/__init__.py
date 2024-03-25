import socket
from pathlib import Path

TEXT_COLUMN = "text"
FILENAME_COLUMN = "filename"
SEED = 666
GPU_SERVERS = ["groesplu0", "GROESSLAO01"]
GPU_DSVD = ["pf9sodsia001", "pf9sodsia002", "pf9sodsia003"]
BASE_DATA_DIR = (
    Path("/data/weak_signals/data/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/nlp/data/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "data"
)
