import socket
from pathlib import Path

GPU_SERVERS = ["groesplu0", "GROESSLAO01"]

FEED_BASE_DIR = (
    Path("/data/weak_signals/data/bertopic/feeds/")
    if socket.gethostname() in GPU_SERVERS
    else Path(__file__).parent.parent.parent / "data" / "bertopic" / "feeds"
)
