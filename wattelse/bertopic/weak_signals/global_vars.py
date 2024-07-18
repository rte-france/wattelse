import os
import json
import socket
from pathlib import Path
import streamlit as st
from loguru import logger
TEXT_COLUMN = "text"
FILENAME_COLUMN = "filename"
SEED = 666
GPU_SERVERS = ["groesplu0", "GROESSLAO01"]
GPU_DSVD = ["pf9sodsia001"]



BASE_DATA_DIR = (
    Path("/data/weak_signals/data/bertopic/Big Datasets/")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/weak_signals/data/")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent / "data"
)

DATA_PATH = BASE_DATA_DIR.absolute().as_posix() +'/'
# DATA_PATH = Path(cwd) / "data" / "bertopic"

STOP_WORDS_RTE = ["w", "kw", "mw", "gw", "tw", "wh", "kwh", "mwh", "gwh", "twh", "volt", "volts", "000"]
COMMON_NGRAMS = [
    "éléctricité",
    "RTE",
    "France",
    "électrique",
    "projet",
    "année",
    "transport électricité",
    "réseau électrique",
    "gestionnaire réseau",
    "réseau transport",
    "production électricité",
    "milliards euros",
    "euros",
    "2022",
    "2023",
    "2024",
    "électricité RTE",
    "Réseau transport",
    "RTE gestionnaire",
    "électricité France",
    "système électrique"
]

# Define the path to your JSON file
stopwords_fr_file = Path(__file__).parent / 'stopwords-fr.json'

# Read the JSON data from the file and directly assign it to the list
with open(stopwords_fr_file, 'r', encoding='utf-8') as file:
    FRENCH_STOPWORDS = json.load(file)