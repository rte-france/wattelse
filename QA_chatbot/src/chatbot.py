import pdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import make_docs_embedding, extract_n_most_relevent_extracts, generate_answer


### Parameters ###

pdf_path = "./data/BP-2019.csv"
embedding_model_name = "dangvantuan/sentence-camembert-large"
instruct_model_name = "bofenghuang/vigogne-2-7b-instruct"
n = 5 # number of top relevent extracts to include as context in the prompt


### Load models ###

embedding_model = SentenceTransformer(embedding_model_name)
tokenizer = AutoTokenizer.from_pretrained(instruct_model_name, padding_side="right", use_fast=False)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_name, torch_dtype=torch.float16, device_map="auto")


### Load data ###

data = pd.read_csv(pdf_path)
docs = data["text"]
docs_embeddings = make_docs_embedding(docs, embedding_model)



### Chatbot ###

while True:
    query = input("Question :")
    relevent_extracts = extract_n_most_relevent_extracts(n, query, docs, docs_embeddings, embedding_model)
    generate_answer(instruct_model, tokenizer, query, relevent_extracts)
    