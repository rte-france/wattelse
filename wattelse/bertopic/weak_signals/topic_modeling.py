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
import streamlit as st

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
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True),
        representation_model=mmr_model,
        # top_n_words=top_n_words,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
        ).fit(docs, embeddings)

    return topic_model

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



def train_topic_models(grouped_data: Dict[pd.Timestamp, pd.DataFrame],
                       embedding_model: SentenceTransformer,
                       embeddings: np.ndarray,
                       umap_model: UMAP,
                       hdbscan_model: HDBSCAN,
                       vectorizer_model: CountVectorizer,
                       mmr_model,
                       top_n_words: int,
                       zeroshot_topic_list: List[str],
                       zeroshot_min_similarity: float) -> Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
    """
    Train BERTopic models for each timestamp in the grouped data.

    Args:
        grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
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

    for i, (period, group) in enumerate(non_empty_groups):
        docs = group['text'].tolist()
        embeddings_subset = embeddings[group.index]

        try:
            topic_model = create_topic_model(docs, embedding_model, embeddings_subset, umap_model, hdbscan_model, vectorizer_model, mmr_model, top_n_words, zeroshot_topic_list, zeroshot_min_similarity)

            doc_info_df = topic_model.get_document_info(docs=docs)
            doc_info_df = doc_info_df.rename(columns={"Document": "Paragraph"})
            doc_info_df = doc_info_df.merge(group[['text', 'document_id', 'source', 'url']], left_on='Paragraph', right_on='text', how='left')
            doc_info_df = doc_info_df.drop(columns=['text'])

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

            # For debug purposes, every 5 iterations print the list of topics obtained
            # if i % 5 == 0:
            #     logger.debug(f"List of topics obtained at {period}: {topic_info_df['Representation'].tolist()}")

        except Exception as e:
            logger.debug(f"{e}")
            logger.debug(f"There isn't enough data in the dataframe corresponding to the period {period}. Skipping...")
            continue
        
        # Update progress bar
        progress = (i + 1) / len(non_empty_groups)
        progress_bar.progress(progress)
        progress_text.text(f"Training BERTopic model for {period} ({i+1}/{len(non_empty_groups)})")

    return topic_models, doc_groups, emb_groups




# def merge_models_old(df1: pd.DataFrame, df2: pd.DataFrame, min_similarity: float, timestamp: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Merge two BERTopic models by comparing topic embeddings and combining similar topics.

#     Args:
#         df1 (pd.DataFrame): DataFrame with topic information from the first model.
#         df2 (pd.DataFrame): DataFrame with topic information from the second model.
#         min_similarity (float): Minimum similarity threshold for merging topics.
#         timestamp (pd.Timestamp): Timestamp of the merge operation.

#     Returns:
#         Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#             - merged_df: DataFrame with merged topic information.
#             - merge_history_df: DataFrame with merge history.
#             - new_topics_df: DataFrame with new topics added from the second model.
#     """
#     merged_df = df1.copy()
#     merge_history = []
#     new_topics = []

#     embeddings1 = np.stack(df1['Embedding'].values)
#     embeddings2 = np.stack(df2['Embedding'].values)

#     similarities = cosine_similarity(embeddings1, embeddings2)
#     max_similarities = np.max(similarities, axis=0)
#     max_similar_topics = df1['Topic'].values[np.argmax(similarities, axis=0)]

#     new_topics_mask = max_similarities < min_similarity
#     new_topics_data = df2[new_topics_mask].copy()
#     new_topics_data['Topic'] = np.arange(merged_df['Topic'].max() + 1, merged_df['Topic'].max() + 1 + len(new_topics_data))
#     new_topics_data['Timestamp'] = timestamp

#     merged_df = pd.concat([merged_df, new_topics_data], ignore_index=True)
#     new_topics = new_topics_data.copy()

#     merge_topics_mask = ~new_topics_mask
#     for row in df2[merge_topics_mask].itertuples(index=False):
#         topic2, count2, doc_count2, representation2, documents2, embedding2, doc_embeddings2, source2, urls2 = row
#         max_similar_topic = max_similar_topics[topic2]
#         similar_row = df1[df1['Topic'] == max_similar_topic].iloc[0]
#         count1 = similar_row['Count']
#         doc_count1 = similar_row['Document_Count']
#         documents1 = similar_row['Documents']
#         source1 = similar_row['Sources']
#         urls1 = similar_row['URLs']

#         merged_count = merged_df.loc[merged_df['Topic'] == max_similar_topic, 'Count'].values[0] + count2
#         merged_doc_count = merged_df.loc[merged_df['Topic'] == max_similar_topic, 'Document_Count'].values[0] + doc_count2
#         merged_documents = merged_df.loc[merged_df['Topic'] == max_similar_topic, 'Documents'].values[0] + documents2
#         merged_source = merged_df.loc[merged_df['Topic'] == max_similar_topic, 'Sources'].values[0] + source2
#         merged_urls = merged_df.loc[merged_df['Topic'] == max_similar_topic, 'URLs'].values[0] + urls2

#         index = merged_df[merged_df['Topic'] == max_similar_topic].index[0]
#         merged_df.at[index, 'Count'] = merged_count
#         merged_df.at[index, 'Document_Count'] = merged_doc_count
#         merged_df.at[index, 'Documents'] = merged_documents
#         merged_df.at[index, 'Sources'] = merged_source
#         merged_df.at[index, 'Embedding'] = similar_row['Embedding']
#         merged_df.at[index, 'URLs'] = merged_urls

#         merge_history.append({
#             'Timestamp': timestamp,
#             'Topic1': max_similar_topic,
#             'Topic2': topic2,
#             'Representation1': similar_row['Representation'],
#             'Representation2': representation2,
#             'Embedding1': similar_row['Embedding'],
#             'Embedding2': embedding2,
#             'Similarity': max_similarities[topic2],
#             'Count1': count1,
#             'Count2': count2,
#             'Document_Count1': doc_count1,
#             'Document_Count2': doc_count2,
#             'Documents1': documents1,
#             'Documents2': documents2,
#             'Source1': source1,
#             'Source2': source2,
#             'URLs1': urls1,
#             'URLs2': urls2,
#         })

#     return merged_df, pd.DataFrame(merge_history), new_topics