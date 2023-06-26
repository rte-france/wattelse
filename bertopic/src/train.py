import pdb

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.backend import BaseEmbedder


from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import torch
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



# Parameters:

data_path = "./data/cleaned/covid_news.csv"
model_name = "paraphrase-multilingual-MiniLM-L12-v2"


# Step 1 - Embedding model
class EmbeddingModel(BaseEmbedder):
    """
    Custom class for the embedding model. Currently supports SentenceBert models (model_name should refer to a SentenceBert model).
    """
    def __init__(self, model_name):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.info(f"Loading model: {model_name} on device: {device}")
        self.embedding_model = SentenceTransformer(model_name)
        logging.info("Model loaded")
        

    def embed(self, documents, verbose=True):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings 

embedding_model = EmbeddingModel(model_name)


# Step 2 - Dimensionality reduction algorithm
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


# Step 3 - Clustering algorithm
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


# Step 4 - Count vectorizer
vectorizer_model = CountVectorizer(stop_words=stopwords.words('french'))


# Step 5 - c-TF-IDF model
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)


# Build BERTopic model
topic_model = BERTopic(
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  ctfidf_model=ctfidf_model,
  top_n_words=10,
  n_gram_range=(1,1),
  nr_topics=10,
  calculate_probabilities=True,
  verbose=True,
)


# Load data and train BERTopic

logging.info(f"Loading data from: {data_path}...")
df = pd.read_csv(data_path)
logging.info("Data loaded")

docs = df["maintext"]
timestamps = df["date_publish"]


logging.info("Fitting BERTopic...")
topics, probs = topic_model.fit_transform(docs)
logging.info("Model fitted")

pdb.set_trace()