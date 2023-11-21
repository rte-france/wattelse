import socket
from pathlib import Path

SEED = 666

GPU_SERVERS = ["groesplu0", "GROESSLAO01"]
GPU_DSVD = ["pf9sodsia001"]
BASE_DATA_DIR = (
    Path("/data/weak_signals/data/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/data/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "data"
)

BASE_OUTPUT_DIR = (
    Path("/data/weak_signals/output/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/output/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "output"
)

BASE_CACHE_PATH = (
    Path("/data/weak_signals/cache/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/cache/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "cache"
)

MODEL_BASE_PATH = (
    Path("/data/weak_signals/models/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/models/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "models"
)

FEED_BASE_DIR = (
    Path("/data/weak_signals/data/bertopic/feeds/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/data/bertopic/feeds/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "data" / "bertopic" / "feeds"
)

LOG_DIR = (
    Path("/data/weak_signals/log/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/log/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "log"
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
