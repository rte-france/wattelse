import socket
from pathlib import Path

GPU_SERVERS = ["groesplu0", "GROESSLAO01"]

BASE_DATA_DIR = (
    Path("/data/weak_signals/data/")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "data"
)

FEED_BASE_DIR = (
    Path("/data/weak_signals/data/bertopic/feeds/")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "data" / "bertopic" / "feeds"
)

LOG_DIR = (
    Path("/data/weak_signals/log/")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "log"
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
