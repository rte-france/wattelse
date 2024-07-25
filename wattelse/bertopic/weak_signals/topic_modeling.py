from typing import List, Dict, Tuple
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from loguru import logger
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from global_vars import *
import streamlit as st
import torch
from tqdm import tqdm
from wattelse.bertopic.utils import TEXT_COLUMN

def create_topic_model(docs: List[str], 
                       embedding_model: SentenceTransformer, 
                       embeddings: np.ndarray, 
                       umap_model: UMAP, 
                       hdbscan_model: HDBSCAN, 
                       vectorizer_model: CountVectorizer, 
                       mmr_model: MaximalMarginalRelevance, 
                       top_n_words: int, 
                       zeroshot_topic_list: List[str], 
                       zeroshot_min_similarity: float) -> BERTopic:
    """
    Create a BERTopic model.

    Args:
        docs (List[str]): List of documents.
        embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
        embeddings (np.ndarray): Precomputed document embeddings.
        umap_model (UMAP): UMAP model for dimensionality reduction.
        hdbscan_model (HDBSCAN): HDBSCAN model for clustering.
        vectorizer_model (CountVectorizer): CountVectorizer model for creating the document-term matrix.
        mmr_model (MaximalMarginalRelevance): MMR model for diverse topic representation.
        top_n_words (int): Number of top words to include in topic representations.
        zeroshot_topic_list (List[str]): List of topics for zero-shot classification.
        zeroshot_min_similarity (float): Minimum similarity threshold for zero-shot classification.

    Returns:
        BERTopic: A fitted BERTopic model.
    """
    logger.debug(f"Creating topic model with zeroshot_topic_list: {zeroshot_topic_list}")
    try:
        # Handle scenario where user enters a bunch of white space characters or any scenario where we can't extract zeroshot topics
        # BERTopic needs a "None" instead of an empty list, otherwise it'll attempt zeroshot topic modeling on an empty list
        if len(zeroshot_topic_list) == 0:
            zeroshot_topic_list = None
        
        logger.debug("\tInitializing BERTopic model")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=False)
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=mmr_model,
            zeroshot_topic_list=zeroshot_topic_list,
            zeroshot_min_similarity=zeroshot_min_similarity,
        )
        logger.success("\tBERTopic model instance created successfully")
        
        logger.debug("\tFitting BERTopic model")
        topics, probs = topic_model.fit_transform(docs, embeddings)

        logger.debug("\tReducing outliers")
        new_topics = topic_model.reduce_outliers(documents=docs, 
                                                 topics=topics, 
                                                 embeddings=embeddings, 
                                                 strategy="embeddings")
        
        topic_model.update_topics(docs=docs,
                                  topics=new_topics,
                                  vectorizer_model=vectorizer_model,
                                  ctfidf_model=ctfidf_model,
                                  representation_model=mmr_model)
        

        logger.success("\tBERTopic model fitted successfully")
        
        return topic_model
    except Exception as e:
        logger.error(f"\tError in create_topic_model: {str(e)}")
        logger.exception("\tTraceback:")
        raise

def preprocess_model(topic_model: BERTopic, docs: List[str], embeddings: np.ndarray) -> pd.DataFrame:
    """
    Preprocess a BERTopic model by extracting topic information, document groups, document embeddings, and URLs.

    Args:
        topic_model (BERTopic): A fitted BERTopic model.
        docs (List[str]): List of documents.
        embeddings (np.ndarray): Precomputed document embeddings.

    Returns:
        pd.DataFrame: A DataFrame with topic information, document groups, document embeddings, and URLs.
    """
    topic_info = topic_model.topic_info_df
    doc_info = topic_model.doc_info_df
    doc_groups = doc_info.groupby('Topic')['Paragraph'].apply(list)

    topic_doc_embeddings = []
    topic_embeddings = []
    topic_sources = []
    topic_urls = []

    for topic_docs in doc_groups:
        doc_embeddings = [embeddings[docs.index(doc)] for doc in topic_docs]
        topic_doc_embeddings.append(doc_embeddings)
        topic_embeddings.append(np.mean(doc_embeddings, axis=0))
        topic_sources.append(doc_info[doc_info['Paragraph'].isin(topic_docs)]['source'].tolist())
        topic_urls.append(doc_info[doc_info['Paragraph'].isin(topic_docs)]['url'].tolist())

    topic_df = pd.DataFrame({
        'Topic': topic_info['Topic'],
        'Count': topic_info['Count'],
        'Document_Count': topic_info['Document_Count'],
        'Representation': topic_info['Representation'],
        'Documents': doc_groups.tolist(),
        'Embedding': topic_embeddings,
        'DocEmbeddings': topic_doc_embeddings,
        'Sources': topic_sources,
        'URLs': topic_urls
    })

    return topic_df

def merge_models(df1: pd.DataFrame, df2: pd.DataFrame, min_similarity: float, timestamp: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = df1.copy()
    merge_history = []
    new_topics = []

    embeddings1 = np.stack(df1['Embedding'].values)
    embeddings2 = np.stack(df2['Embedding'].values)

    similarities = cosine_similarity(embeddings1, embeddings2)
    max_similarities = np.max(similarities, axis=0)
    max_similar_topics = df1['Topic'].values[np.argmax(similarities, axis=0)]

    new_topics_mask = max_similarities < min_similarity
    new_topics_data = df2[new_topics_mask].copy()
    new_topics_data['Topic'] = np.arange(merged_df['Topic'].max() + 1, merged_df['Topic'].max() + 1 + len(new_topics_data))
    new_topics_data['Timestamp'] = timestamp

    merged_df = pd.concat([merged_df, new_topics_data], ignore_index=True)
    new_topics = new_topics_data.copy()

    merge_topics_mask = max_similarities >= min_similarity
    merge_topics_data = df2[merge_topics_mask]

    for max_similar_topic, group in merge_topics_data.groupby(max_similar_topics[merge_topics_mask]):
        similar_row = df1[df1['Topic'] == max_similar_topic].iloc[0]
        index = merged_df[merged_df['Topic'] == max_similar_topic].index[0]

        merged_df.at[index, 'Count'] += group['Count'].sum()
        merged_df.at[index, 'Document_Count'] += group['Document_Count'].sum()
        
        # Update the 'Documents' field with only the new documents from the current timestamp
        new_documents = [doc for docs in group['Documents'] for doc in docs]
        merged_df.at[index, 'Documents'] = similar_row['Documents'] + [(timestamp, new_documents)]
        
        merged_df.at[index, 'Sources'] += [source for sources in group['Sources'] for source in sources]
        merged_df.at[index, 'URLs'] += [url for urls in group['URLs'] for url in urls]

        merge_history.extend({
            'Timestamp': timestamp,
            'Topic1': max_similar_topic,
            'Topic2': row['Topic'],
            'Representation1': similar_row['Representation'],
            'Representation2': row['Representation'],
            'Embedding1': similar_row['Embedding'],
            'Embedding2': row['Embedding'],
            'Similarity': max_similarities[row['Topic']],
            'Count1': len(similar_row['Documents']),
            'Count2': len(row['Documents']),
            'Document_Count1': similar_row['Document_Count'],
            'Document_Count2': row['Document_Count'],
            'Documents1': similar_row['Documents'],
            'Documents2': row['Documents'],
            'Source1': similar_row['Sources'],
            'Source2': row['Sources'],
            'URLs1': similar_row['URLs'],
            'URLs2': row['URLs'],
        } for _, row in group.iterrows())

    return merged_df, pd.DataFrame(merge_history), new_topics


def embed_documents(texts: List[str], 
                    embedding_model_name: str, 
                    embedding_dtype: str, 
                    embedding_device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    batch_size: int = 5000,
                    max_seq_length: int = 512) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Embed a list of documents using a Sentence Transformer model.

    This function loads a specified Sentence Transformer model and uses it to create
    embeddings for a list of input texts. It processes the texts in batches to manage
    memory efficiently, especially for large datasets.

    Args:
        texts (List[str]): A list of text documents to be embedded.
        embedding_model_name (str): The name of the Sentence Transformer model to use.
        embedding_dtype (str): The data type to use for the embeddings ('float32', 'float16', or 'bfloat16').
        embedding_device (str, optional): The device to use for embedding ('cuda' or 'cpu'). 
                                          Defaults to 'cuda' if available, else 'cpu'.
        batch_size (int, optional): The number of texts to process in each batch. Defaults to 32.
        max_seq_length (int, optional): The maximum sequence length for the model. Defaults to 512.

    Returns:
        Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
            - The loaded and configured Sentence Transformer model.
            - A numpy array of embeddings, where each row corresponds to a text in the input list.

    Raises:
        ValueError: If an invalid embedding_dtype is provided.
    """
    # Configure model kwargs based on the specified dtype
    model_kwargs = {}
    if embedding_dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif embedding_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif embedding_dtype != "float32":
        raise ValueError("Invalid embedding_dtype. Must be 'float32', 'float16', or 'bfloat16'.")

    # Load the embedding model
    embedding_model = SentenceTransformer(embedding_model_name, 
                                          device=embedding_device, 
                                          trust_remote_code=True, 
                                          model_kwargs=model_kwargs)
    embedding_model.max_seq_length = max_seq_length

    # Calculate the number of batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    # Initialize an empty list to store embeddings
    embeddings = []

    # Process texts in batches
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    return embedding_model, embeddings


def train_topic_models(grouped_data: Dict[pd.Timestamp, pd.DataFrame],
                       embedding_model: SentenceTransformer,
                       embeddings: np.ndarray,
                       umap_n_components: int,
                       umap_n_neighbors: int,
                       hdbscan_min_cluster_size: int,
                       hdbscan_min_samples: int,
                       hdbscan_cluster_selection_method: str,
                       vectorizer_ngram_range: Tuple[int, int],
                       min_df: int,
                       top_n_words: int,
                       zeroshot_topic_list: List[str],
                       zeroshot_min_similarity: float,
                       language: str) -> Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
    """
    Train BERTopic models for each timestamp in the grouped data.

    Args:
        grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
        embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
        embeddings (np.ndarray): Precomputed document embeddings.
        umap_n_components (int): Number of components for UMAP.
        umap_n_neighbors (int): Number of neighbors for UMAP.
        hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN.
        hdbscan_min_samples (int): Minimum samples for HDBSCAN.
        hdbscan_cluster_selection_method (str): Cluster selection method for HDBSCAN.
        vectorizer_ngram_range (Tuple[int, int]): N-gram range for CountVectorizer.
        min_df (int): Minimum document frequency for CountVectorizer.
        top_n_words (int): Number of top words to include in topic representations.
        zeroshot_topic_list (List[str]): List of topics for zero-shot classification.
        zeroshot_min_similarity (float): Minimum similarity threshold for zero-shot classification.
        language (str): Used to determine the list of stopwords to use
        
    Returns:
        Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
            - topic_models: Dictionary of trained BERTopic models for each timestamp.
            - doc_groups: Dictionary of document groups for each timestamp.
            - emb_groups: Dictionary of document embeddings for each timestamp.
    """
    topic_models = {}
    doc_groups = {}
    emb_groups = {}

    non_empty_groups = [(period, group) for period, group in grouped_data.items() if not group.empty]

    # Set up progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    logger.debug(f"Starting to train topic models with zeroshot_topic_list: {zeroshot_topic_list}")

    for i, (period, group) in enumerate(non_empty_groups):
        docs = group[TEXT_COLUMN].tolist()
        embeddings_subset = embeddings[group.index]

        logger.debug(f"Processing period: {period}")
        logger.debug(f"Number of documents: {len(docs)}")

        try:
            umap_model = UMAP(n_components=umap_n_components, n_neighbors=umap_n_neighbors, min_dist=DEFAULT_UMAP_MIN_DIST, random_state=42, metric="cosine")
            hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric='euclidean', cluster_selection_method=hdbscan_cluster_selection_method, prediction_data=True)
            
            stopword_set = STOPWORDS if language=="French" else "english"
            vectorizer_model = CountVectorizer(stop_words=stopword_set, min_df=min_df, ngram_range=vectorizer_ngram_range)
            mmr_model = MaximalMarginalRelevance(diversity=DEFAULT_MMR_DIVERSITY)

            logger.debug("Creating topic model...")
            topic_model = create_topic_model(docs, 
                                             embedding_model, 
                                             embeddings_subset, 
                                             umap_model, 
                                             hdbscan_model, 
                                             vectorizer_model, 
                                             mmr_model, 
                                             top_n_words, 
                                             zeroshot_topic_list, 
                                             zeroshot_min_similarity)
            
            logger.debug("Topic model created successfully")
            
            doc_info_df = topic_model.get_document_info(docs=docs)
            doc_info_df = doc_info_df.rename(columns={"Document": "Paragraph"})
            doc_info_df = doc_info_df.merge(group[[TEXT_COLUMN, 'document_id', 'source', 'url']], left_on='Paragraph', right_on=TEXT_COLUMN, how='left')
            doc_info_df = doc_info_df.drop(columns=[TEXT_COLUMN])

            topic_info_df = topic_model.get_topic_info()
            topic_doc_count_df = doc_info_df.groupby('Topic')['document_id'].nunique().reset_index(name='Document_Count')
            topic_sources_df = doc_info_df.groupby('Topic')['source'].apply(list).reset_index(name='Sources')
            topic_urls_df = doc_info_df.groupby('Topic')['url'].apply(list).reset_index(name='URLs')

            topic_info_df = topic_info_df.merge(topic_doc_count_df, on='Topic', how='left')
            topic_info_df = topic_info_df.merge(topic_sources_df, on='Topic', how='left')
            topic_info_df = topic_info_df.merge(topic_urls_df, on='Topic', how='left')

            topic_info_df = topic_info_df[['Topic', 'Count', 'Document_Count', 'Representation', 'Name', 'Representative_Docs', 'Sources', 'URLs']]

            topic_model.doc_info_df = doc_info_df
            topic_model.topic_info_df = topic_info_df

            topic_models[period] = topic_model
            doc_groups[period] = docs
            emb_groups[period] = embeddings_subset

            logger.debug(f"Successfully processed period: {period}")

        except Exception as e:
            logger.error(f"Error processing period {period}: {str(e)}")
            logger.exception("Traceback:")
            continue
        
        # Update progress bar
        progress = (i + 1) / len(non_empty_groups)
        progress_bar.progress(progress)
        progress_text.text(f"Training BERTopic model for {period} ({i+1}/{len(non_empty_groups)})")

    logger.debug("Finished training all topic models")
    return topic_models, doc_groups, emb_groups