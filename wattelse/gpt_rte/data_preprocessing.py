import pdb

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


# Parameters

df_path = "./data/raw_data/origami_dataset_for_LM_v2023.hdf"
tokenizer_name = "asi/gpt-fr-cased-small"
series_paragraphs_save_path = "./data/processed_data/series_tokenized_paragraphs_"+tokenizer_name.replace("/","-")+".pickle"



# Import raw data

df = pd.read_hdf(df_path)


# Clean data

df = df[df["text"] != ""].reset_index(drop=True)
logging.info(f"Number of documents: {len(df)}")

# Extract every paragraphs of every document

concatenated_texts = "\n\n\n".join(df["text"]) # concatenate every documents in a single variable
paragraphs = concatenated_texts.split("\n\n\n") # split paragraphs
paragraphs = [elem for elem in paragraphs if elem != ""] # remove empty paragraphs
paragraphs = [elem for elem in paragraphs if len(elem)>50] # keep only paragraphs with more than 50 characters

# Make paragraphs DataFrame and clean it

df_paragraphs = pd.DataFrame(paragraphs, columns=["paragraphs"])
df_paragraphs["paragraphs"] = df_paragraphs["paragraphs"].apply(lambda text: re.sub("\n\n", "\n", text))
df_paragraphs["paragraphs"] = df_paragraphs["paragraphs"].apply(lambda text: re.sub("^\n", "", text))
logging.info(f"Total number of extracted paragraphs: {len(df_paragraphs)}")


# Tokenize documents

logging.info("Tokenizing paragraphs using: "+tokenizer_name+"...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
df_paragraphs["tokenized_paragraphs"] = df_paragraphs.apply(lambda row: tokenizer(row.paragraphs), axis=1)
logging.info("Paragraphs tokenized")


# Plot stats

df_paragraphs["n_tokens"] = df_paragraphs.apply(lambda row: len(row.tokenized_paragraphs["input_ids"]), axis=1)

sns.set_theme()

plt.figure()
hist_plot = sns.displot(data=df_paragraphs, x="n_tokens", kde=True, log_scale=True)
hist_plot.savefig("./figures/hist_n_tokens_paragraphs_"+tokenizer_name.replace("/","-"), bbox_inches='tight')

logging.info("Plots saved")


# Save DataFrame

df_paragraphs["tokenized_paragraphs"].to_pickle(series_paragraphs_save_path)
logging.info("Tokenized DataFrame saved at: " + series_paragraphs_save_path)