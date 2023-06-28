import nltk

nltk.download("stopwords")
import pandas as pd
import jsonlines

DATA_DIR = "./data/"
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"


def file_to_pd(data_name: str, base_dir: str = None) -> pd.DataFrame:
    data_path = base_dir + data_name if base_dir else data_name
    if ".csv" in data_name:
        return pd.read_csv(data_path)
    elif ".jsonl" in data_name or ".jsonlines" in data_name:
        with open(data_path, "r") as f:
            with jsonlines.Reader(f) as reader:
                data = reader.read()
                return pd.DataFrame(data)
