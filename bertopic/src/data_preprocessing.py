import pdb

import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Parameters

data_path = "./data/raw/curebot.jsonlines"
data_save_path = "./data/cleaned/curebot_1200_news.csv"


# Load and clean data

# logging.info(f"Loading data from: {data_path}...")
# df = pd.read_csv(data_path).dropna().reset_index(drop=True)
# logging.info(f"Data loaded, {len(df)} rows")

# df = df[(df["date_publish"]>"2020-01-01 00:00:00") & (df["date_publish"]<"2024-01-01 00:00:00")].reset_index(drop=True)
# df = df.rename(columns={'maintext': 'docs', 'date_publish': 'timestamps'})
# logging.info(f"Data cleaned, {len(df)} rows")


# # Save cleaned DataFrame

# df.to_csv(data_save_path)
# logging.info(f"Saved data to: {data_save_path}")

df = pd.read_json(data_path)

df = df[df["text"]!=""].reset_index(drop=True)

df = df.rename(columns={'text': 'docs', 'timestamp': 'timestamps'})

df.to_csv(data_save_path)