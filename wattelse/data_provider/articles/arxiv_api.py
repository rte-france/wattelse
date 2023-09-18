import pdb

import arxiv
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)


### Request parameters ###

query = "cat:cs.CL"
max_results = float("inf")
save_path = "./data/arxiv_nlp_papers.csv"

custom_client = arxiv.Client(
    page_size = 2000,
    delay_seconds = 3,
    num_retries = 10,
    )


### API call ###

search = custom_client.results(
    arxiv.Search(
        query = query,
        max_results=max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending,
	)
)


### Save to pandas Dataframe ###

df_list = []

for paper in search:
    df_list.append([paper.entry_id, paper.title, paper.summary, paper.published])

df = pd.DataFrame(df_list, columns =["id", "title", "text", "timestamp"])
df = df.dropna()
df.to_csv(save_path)
logging.info(f"Saved new DF to: {save_path}")