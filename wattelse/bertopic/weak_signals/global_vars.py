#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import os
from pathlib import Path

from wattelse.common import BASE_CACHE_PATH, BASE_DATA_PATH

# Stopwords
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

stopwords_fr_file = Path(__file__).parent / 'stopwords-fr.json'
with open(stopwords_fr_file, 'r', encoding='utf-8') as file:
    FRENCH_STOPWORDS = json.load(file)

STOPWORDS = STOP_WORDS_RTE + COMMON_NGRAMS + FRENCH_STOPWORDS

# Paths
CACHE_PATH = BASE_CACHE_PATH / "weak_signals"
MODELS_DIR = CACHE_PATH / "models"
DATA_PATH = BASE_DATA_PATH / "bertopic"
ZEROSHOT_TOPICS_DATA_DIR = CACHE_PATH / "zeroshot_topics_data"
SIGNAL_EVOLUTION_DATA_DIR = CACHE_PATH / "signal_evolution_data"

# File names
STATE_FILE = 'app_state.pkl'
EMBEDDINGS_FILE = 'embeddings.npy'
DOC_GROUPS_FILE = 'doc_groups.pkl'
EMB_GROUPS_FILE = 'emb_groups.pkl'
GRANULARITY_FILE = 'granularity.pkl'
HYPERPARAMS_FILE = 'hyperparams.pkl'
DOC_INFO_DF_FILE = 'doc_info_df.pkl'
TOPIC_INFO_DF_FILE = 'topic_info_df.pkl'
MODELS_TRAINED_FILE = 'models_trained_flag.pkl'

# Model file names
ZEROSHOT_TOPICS_DATA_FILE = 'zeroshot_topics_data.json'
INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE = 'individual_topic_counts.json'
CUMULATIVE_MERGED_TOPIC_COUNTS_FILE = 'cumulative_topic_counts.json'

# Embedding models
ENGLISH_EMBEDDING_MODELS = ["all-mpnet-base-v2", 
                            "Alibaba-NLP/gte-base-en-v1.5", 
                            "all-MiniLM-L12-v2"]
FRENCH_EMBEDDING_MODELS = ["OrdalieTech/Solon-embeddings-base-0.1", 
                           "OrdalieTech/Solon-embeddings-large-0.1", 
                           "dangvantuan/sentence-camembert-large", 
                           "antoinelouis/biencoder-distilcamembert-mmarcoFR"]

# BERTopic Hyperparameters
DEFAULT_UMAP_N_COMPONENTS = 5
DEFAULT_UMAP_N_NEIGHBORS = 5
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5
DEFAULT_HDBSCAN_MIN_SAMPLES = 5
DEFAULT_TOP_N_WORDS = 10
DEFAULT_MIN_DF = 1
DEFAULT_GRANULARITY = 2
DEFAULT_MIN_SIMILARITY = 0.7
DEFAULT_ZEROSHOT_MIN_SIMILARITY = 0.5
BERTOPIC_SERIALIZATION = "safetensors" # or pickle
DEFAULT_MMR_DIVERSITY = 0.3
DEFAULT_UMAP_MIN_DIST = 0.0
OUTLIER_REDUCTION_STRATEGY = "c-tf-idf" # or "embeddings"

# Embedding Settings
EMBEDDING_DTYPES = ["float32", "float16", "bfloat16"]
EMBEDDING_BATCH_SIZE = 5000
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_DEVICE = "cuda"

# Other constants
LANGUAGES = ["French", "English"]
HDBSCAN_CLUSTER_SELECTION_METHODS = ["eom", "leaf"]
VECTORIZER_NGRAM_RANGES = [(1, 2), (1, 1), (2, 2)]

# GPT Model Settings
GPT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL_NAME", "gpt-4o-mini")
GPT_TEMPERATURE = 0.0
GPT_SYSTEM_MESSAGE = "You are a helpful assistant, skilled in detailing topic evolution over time for the detection of emerging trends and signals."

# Data Processing
MIN_CHARS_DEFAULT = 100
SAMPLE_SIZE_DEFAULT = None  # Or whatever default you want, None means all documents

# Time Settings
DEFAULT_WINDOW_SIZE = 7  # days
MAX_WINDOW_SIZE = 365  # days

# UI Settings
PAGE_TITLE = "BERTopic Topic Detection"
LAYOUT = "wide"

# Visualization Settings
SANKEY_NODE_PAD = 15
SANKEY_NODE_THICKNESS = 20
SANKEY_LINE_COLOR = "black"
SANKEY_LINE_WIDTH = 0.5

# Data Analysis Settings
POPULARITY_THRESHOLD = 0.1  # for weak signal detection, if applicable

# Signal classification Settings
SIGNAL_CLASSIF_LOWER_BOUND = 10
SIGNAL_CLASSIF_UPPER_BOUND = 75

# Error Messages
NO_DATA_WARNING = "No data available for the selected granularity."
NO_MODELS_WARNING = "No saved models found."
NO_CACHE_WARNING = "No cache found."
TOPIC_NOT_FOUND_WARNING = "Topic {topic_number} not found in the merge histories within the specified window."

# Success Messages
STATE_SAVED_MESSAGE = "Application state saved."
STATE_RESTORED_MESSAGE = "Application state restored."
MODELS_SAVED_MESSAGE = "Models saved."
MODELS_RESTORED_MESSAGE = "Models restored."
EMBEDDINGS_CALCULATED_MESSAGE = "Embeddings calculated successfully!"
MODEL_TRAINING_COMPLETE_MESSAGE = "Model training complete!"
MODEL_MERGING_COMPLETE_MESSAGE = "Model merging complete!"
TOPIC_COUNTS_SAVED_MESSAGE = "Topic and signal counts saved to {file_path}"
CACHE_PURGED_MESSAGE = "Cache purged."

# Other Constants
DEFAULT_ZEROSHOT_TOPICS = ""  # Empty string or a default list of topics
PROGRESS_BAR_DESCRIPTION = "Batches processed"

