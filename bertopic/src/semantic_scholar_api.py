import pdb

import requests
import pandas as pd
import time

import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

### Parameters ###

arxiv_df_path = "./data/arxiv_nlp_papers_from_2010.csv"
semantic_scholar_df_save_path = arxiv_df_path.replace("arxiv", "semantic_scholar")
api_key = "q252gwgpRVAb49CWJqde3TEr6VjzXyc8P8RcGCo2"
papers_per_request = 500 # should not be more than 500
delay_seconds = 1 # delay between each request
fields = "title,abstract,citationCount,publicationDate,url"


### Call Semantic Scholar API ###

# Load DataFrame and initialize new DF
df = pd.read_csv(arxiv_df_path)

semantic_scholar_items_list = []

# Request papers based on ArXiv ID
for i in range(len(df)//papers_per_request):
	logging.info(f"Requests : {(i+1)*papers_per_request}/{len(df)}")
	ids_list = ("URL:"+df["id"]).iloc[i*papers_per_request:(i+1)*papers_per_request].astype(str).values.tolist()
	r = requests.post(
		"https://api.semanticscholar.org/graph/v1/paper/batch",
		params={"fields": fields},
		json={"ids": ids_list},
		headers={"x-api-key": api_key},
		)
	semantic_scholar_items_list += [item for item in r.json() if item is not None]

	time.sleep(delay_seconds)

# Request last rows
logging.info(f"Requests : {len(df)}/{len(df)}")
ids_list = ("URL:"+df["id"]).iloc[(i+1)*papers_per_request:].astype(str).values.tolist()
r = requests.post(
		"https://api.semanticscholar.org/graph/v1/paper/batch",
		params={"fields": fields},
		json={"ids": ids_list},
		headers={"x-api-key": api_key},
	)
semantic_scholar_items_list += [item for item in r.json() if item is not None]

# Create new DF end save

new_df  = pd.DataFrame(semantic_scholar_items_list)
new_df = new_df.dropna()
new_df.columns = ["id", "url", "title", "text", "citation_count", "timestamp"]
logging.info(f"Retrieved {len(new_df)}/{len(df)} papers")

new_df.to_csv(semantic_scholar_df_save_path)
logging.info(f"Saved new DF to: {semantic_scholar_df_save_path}")
