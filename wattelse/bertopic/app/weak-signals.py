import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import re
import json
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.preprocessing import MinMaxScaler



# Working directory
cwd = os.getcwd()+'/Weak-Signals-Investigations/'

STOP_WORDS_RTE = ["w", "kw", "mw", "gw", "tw", "wh", "kwh", "mwh", "gwh", "twh", "volt", "volts", "000"]


# Define the path to your JSON file
stopwords_fr_file = cwd+'stopwords-fr.json'

# Read the JSON data from the file and directly assign it to the list
with open(stopwords_fr_file, 'r', encoding='utf-8') as file:
    FRENCH_STOPWORDS = json.load(file)


# Set page title
st.set_page_config(page_title="BERTopic Topic Detection", layout="wide")

# Function to create BERTopic models
def create_topic_models(docs, embedding_model, embeddings, umap_model, hdbscan_model, vectorizer_model, mmr_model, top_n_words, zeroshot_topic_list, zeroshot_min_similarity):
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=mmr_model,
        top_n_words=top_n_words,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity
    ).fit(docs, embeddings)
    return topic_model, docs



# Function to look out for weak signals based on the zero-shot list of topics we want to monitor
def detect_weak_signals_zeroshot(topic_models, zeroshot_topic_list):
    weak_signal_trends = {}
    
    for topic in zeroshot_topic_list:
        weak_signal_trends[topic] = {}
        
        for timestamp, topic_model in topic_models.items():
            topic_info = topic_model.get_topic_info()
            
            for _, row in topic_info.iterrows():
                if row['Name'] == topic:
                    weak_signal_trends[topic][timestamp] = {
                        'Representation': row['Representation'],
                        'Representative_Docs': row['Representative_Docs'],
                        'Count': row['Count']
                    }
                    break
    
    return weak_signal_trends

@st.cache_data
def detect_weak_signals(df, metric, num_signals, num_docs, _vectorizer):
    # Calculate coherence scores
    coherence_scores = calculate_coherence_scores(df, _vectorizer, metric)
    
    # Add coherence scores as a new column to the dataframe
    df['Coherence'] = coherence_scores

    # Normalize coherence scores and count values using Min-Max scaling
    scaler = MinMaxScaler()
    df['Normalized_Coherence'] = scaler.fit_transform(df[['Coherence']])
    df['Normalized_Count'] = scaler.fit_transform(df[['Count']])

    # Calculate the weighted combination of normalized coherence and count
    weight_coherence = 0.7  # Adjust the weight given to coherence (e.g., 0.7)
    weight_count = 0.3  # Adjust the weight given to count (e.g., 0.3)
    df['Coherence_Count_Score'] = (weight_coherence * df['Normalized_Coherence']) + (weight_count * df['Normalized_Count'])

    # Select the top topics based on coherence-count score
    top_topics = df.nlargest(num_signals, 'Coherence_Count_Score')

    # Select the bottom topics based on coherence-count score
    bottom_topics = df.nsmallest(num_signals, 'Coherence_Count_Score')

    return top_topics, bottom_topics


def preprocess_model(topic_model, docs, embeddings):
    # Get topic information
    topic_info = topic_model.get_topic_info()
    # Get document information
    doc_info = topic_model.get_document_info(docs)
    # Group documents by topic
    doc_groups = doc_info.groupby('Topic')['Document'].apply(list)
    
    # Create a list to store the document embeddings for each topic
    topic_doc_embeddings = []
    topic_embeddings = []
    
    # Iterate over each topic and retrieve the corresponding document embeddings
    for topic_docs in doc_groups:
        doc_embeddings = [embeddings[docs.index(doc)] for doc in topic_docs]
        topic_doc_embeddings.append(doc_embeddings)
        topic_embeddings.append(np.mean(doc_embeddings, axis=0))
    
    # Create a DataFrame with topic information, document groups, and document embeddings
    topic_df = pd.DataFrame({
        'Topic': topic_info['Topic'],
        'Count': topic_info['Count'],
        'Representation': topic_info['Representation'],
        'Documents': doc_groups.tolist(),
        'Embedding': topic_embeddings,
        'DocEmbeddings': topic_doc_embeddings
    })
    
    return topic_df


def merge_models(df1, df2, min_similarity, timestamp):
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

    merge_topics_mask = ~new_topics_mask
    for topic2, count2, representation2, documents2, embedding2, doc_embeddings2 in df2[merge_topics_mask].itertuples(index=False):
        max_similar_topic = max_similar_topics[topic2]
        similar_row = merged_df[merged_df['Topic'] == max_similar_topic].iloc[0]
        count1 = similar_row['Count']
        documents1 = similar_row['Documents']

        merged_count = count1 + count2
        merged_documents = documents1 + documents2

        index = merged_df[merged_df['Topic'] == max_similar_topic].index[0]
        merged_df.at[index, 'Count'] = merged_count
        merged_df.at[index, 'Documents'] = merged_documents
        merged_df.at[index, 'Embedding'] = similar_row['Embedding']

        merge_history.append({
            'Timestamp': timestamp,
            'Topic1': max_similar_topic,
            'Topic2': topic2,
            'Representation1': similar_row['Representation'],
            'Representation2': representation2,
            'Embedding1': similar_row['Embedding'],
            'Embedding2': embedding2,
            'Similarity': max_similarities[topic2],
            'Count1': count1,
            'Count2': count2
        })

    return merged_df, pd.DataFrame(merge_history), new_topics



def transform_dataframe(df):
    # Create a copy of the original dataframe
    transformed_df = df.copy()
    
    # Create a column 'Timestamp_Index' that maps each timestamp to an index
    timestamps = transformed_df['Timestamp'].unique()
    timestamp_index_map = {timestamp: index for index, timestamp in enumerate(timestamps)}
    transformed_df['Timestamp_Index'] = transformed_df['Timestamp'].map(timestamp_index_map)

    # Group by Topic1 and collect the list of timestamp indices where each Topic1 value appears
    topic1_timestamp_indices = transformed_df.groupby('Topic1')['Timestamp_Index'].apply(list).to_dict()

    # Initialize variables to store the source, destination, representation, timestamp, and count values
    src_values = []
    dest_values = []
    representation_values = []
    timestamp_values = []
    count_values = []

    # Initialize a dictionary to store the mapping of (topic1, timestamp_index) to the new destination topic
    topic1_dest_map = {}

    # Initialize a dictionary to store the mapping of (topic1, timestamp_index) to the merged count
    topic1_count_map = {}

    # Group by Timestamp and process each row
    for timestamp, group in transformed_df.groupby('Timestamp'):
        for _, row in group.iterrows():
            topic1 = row['Topic1']
            topic2 = row['Topic2']
            representation1 = row['Representation1']
            representation2 = row['Representation2']
            timestamp_index = row['Timestamp_Index']
            count1 = row['Count1']
            count2 = row['Count2']
            
            # Generate the source values for Topic1 and Topic2
            src_topic1 = f"{timestamp_index}_1_{topic1}"
            src_topic2 = f"{timestamp_index}_2_{topic2}"
            
            # Check if (topic1, timestamp_index) has a destination topic in the topic1_dest_map
            if (topic1, timestamp_index) in topic1_dest_map:
                dest_topic = topic1_dest_map[(topic1, timestamp_index)]
                
                # Update the merged count for the destination topic
                topic1_count_map[(topic1, timestamp_index)] += count2
                count_merged = topic1_count_map[(topic1, timestamp_index)]
            else:
                # Find the next timestamp index where Topic1 appears
                topic1_future_timestamp_indices = [idx for idx in topic1_timestamp_indices[topic1] if idx > timestamp_index]
                
                if topic1_future_timestamp_indices:
                    next_timestamp_index = topic1_future_timestamp_indices[0]
                    dest_topic = f"{next_timestamp_index}_1_{topic1}"
                else:
                    # If Topic1 doesn't appear in any future timestamps, create a new destination topic
                    dest_topic = f"{timestamp_index}_1_{topic1}_new"
                
                # Store the mapping of (topic1, timestamp_index) to the new destination topic
                topic1_dest_map[(topic1, timestamp_index)] = dest_topic
                
                # Initialize the merged count for the destination topic
                topic1_count_map[(topic1, timestamp_index)] = count1 + count2
                count_merged = topic1_count_map[(topic1, timestamp_index)]
            
            # Determine the timestamp for the current row
            if '_2_' in src_topic2:
                # If the source contains '_2_', find the next available timestamp
                next_timestamp = timestamps[timestamp_index + 1] if timestamp_index + 1 < len(timestamps) else timestamp
            else:
                # Otherwise, use the current timestamp
                next_timestamp = timestamp
            
            # Append the source, destination, representation, timestamp, and count values to the respective lists
            src_values.extend([src_topic1, src_topic2])
            dest_values.extend([dest_topic, dest_topic])
            representation_values.extend([representation1, representation2])
            timestamp_values.extend([timestamp, next_timestamp])
            count_values.extend([count1, count_merged])

    # Create a new dataframe with the source, destination, representation, timestamp, and count values
    transformed_df_new = pd.DataFrame({
        'Source': src_values,
        'Destination': dest_values,
        'Representation': representation_values,
        'Timestamp': timestamp_values,
        'Count': count_values
    })

    # Convert lists to tuples in the 'Representation' column
    transformed_df_new['Representation'] = transformed_df_new['Representation'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Group by Timestamp, Source, and Destination, and keep the row with the smallest Count value for each group
    transformed_df_new = transformed_df_new.loc[transformed_df_new.groupby(['Timestamp', 'Source', 'Destination'])['Count'].idxmin()]

    return transformed_df_new




def create_sankey_diagram(all_merge_histories_df, search_term=None, max_pairs=None):
    # Filter the dataframe based on the search term if provided
    if search_term:
        # Perform recursive search to find connected nodes
        def find_connected_nodes(node, connected_nodes):
            if node not in connected_nodes:
                connected_nodes.add(node)
                connected_df = all_merge_histories_df[(all_merge_histories_df['Source'] == node) | (all_merge_histories_df['Destination'] == node)]
                for _, row in connected_df.iterrows():
                    find_connected_nodes(row['Source'], connected_nodes)
                    find_connected_nodes(row['Destination'], connected_nodes)
        
        # Find nodes that match the search term
        matching_nodes = set(all_merge_histories_df[all_merge_histories_df['Representation'].apply(lambda x: search_term.lower() in str(x).lower())]['Source'])
        
        # Find connected nodes
        connected_nodes = set()
        for node in matching_nodes:
            find_connected_nodes(node, connected_nodes)
        
        # Filter the dataframe based on connected nodes
        all_merge_histories_df = all_merge_histories_df[(all_merge_histories_df['Source'].isin(connected_nodes)) | (all_merge_histories_df['Destination'].isin(connected_nodes))]
    
    # Create nodes and links for the Sankey Diagram
    nodes = []
    links = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for _, row in all_merge_histories_df.iterrows():
        source_node = row['Source']
        target_node = row['Destination']
        timestamp = row['Timestamp']
        representation = ', '.join(row['Representation'])  # Convert tuple to string
        count = row['Count']
        
        # Extract the topic IDs from the source and destination nodes
        source_topic_id = source_node.split('_')[-1]
        target_topic_id = target_node.split('_')[-1]
        
        # Generate label for source node
        source_label = ', '.join(representation.split(', ')[:5])
        
        # Add source node if not already present
        if source_node not in [node['name'] for node in nodes]:
            nodes.append({'name': source_node, 'label': source_label, 'color': colors[len(nodes) % len(colors)]})
        
        # Add target node if not already present
        if target_node not in [node['name'] for node in nodes]:
            nodes.append({'name': target_node, 'color': colors[len(nodes) % len(colors)]})
        
        # Add link between source and target nodes
        link = {
            'source': source_node,
            'target': target_node,
            'value': count,
            'timestamp': timestamp,
            'source_topic_id': source_topic_id,
            'target_topic_id': target_topic_id,
            'representation': representation,
            'count': count
        }
        if link not in links:
            links.append(link)
    
    # Limit the number of pairs displayed based on the max_pairs parameter
    if max_pairs is not None:
        links = links[:max_pairs]
    
    # Create the Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node.get('label', '') for node in nodes],
            color=[node['color'] for node in nodes]
        ),
        link=dict(
            source=[nodes.index(next((node for node in nodes if node['name'] == link['source']), None)) for link in links],
            target=[nodes.index(next((node for node in nodes if node['name'] == link['target']), None)) for link in links],
            value=[link['value'] for link in links],
            hovertemplate='Timestamp: %{customdata[0]}<br>' +
                          'Source Topic ID: %{customdata[1]}<br>' +
                          'Representation: %{customdata[3]}<br>' +
                          'Count: %{customdata[4]}<extra></extra>',
            customdata=[[
                link['timestamp'],
                link['source_topic_id'],
                link['representation'],
                link['count']
            ] for link in links],
            color=[colors[i % len(colors)] for i in range(len(links))]
        ),
        arrangement='snap'
    )])
    
    # Update the layout
    fig.update_layout(
        title_text="Topic Merging Process",
        font_size=15,
        height=1500
    )
    
    return fig




def preprocess_french_text(text):
    # Replace hyphens and similar characters with spaces
    text = re.sub(r'\b(-|/|;|:)', ' ', text)

    # Remove specific prefixes like "l'", "d'", "l’", and "d’" from words
    text = re.sub(r"\b(l'|L'|D'|d'|l’|L’|D’|d’)", '', text)

    # Remove punctuations, excluding apostrophes, hyphens, and importantly, newlines
    # Here, we modify the expression to keep newlines by adding them to the set of allowed characters
    text = re.sub(r'[^\w\s\nàâçéèêëîïôûùüÿñæœ]', '', text)

    # Replace special characters with a space, preserving accented characters, common Latin extensions, and newlines
    # Again, ensure newlines are not removed by including them in the allowed set
    text = re.sub(r'[^\w\s\nàâçéèêëîïôûùüÿñæœ]', ' ', text)

    # Replace multiple spaces with a single space, but do not touch newlines
    # Here, we're careful not to collapse newlines into spaces, so this operation targets spaces only
    text = re.sub(r'[ \t]+', ' ', text)

    return text



def calculate_coherence_scores(df, _vectorizer, metric="c_v"):
    # Preprocess documents
    documents_per_topic = df.groupby(['Topic'], as_index=False).agg({'Documents': lambda x: ' '.join([' '.join(doc) for doc in x])})
    cleaned_docs = documents_per_topic['Documents'].tolist()

    logger.debug("COHERENCE: preprocessing done")

    # Extract analyzer from vectorizer
    analyzer = _vectorizer.build_analyzer()

    # Fit the vectorizer on the entire list of documents
    all_docs = ' '.join(st.session_state.timefiltered_df['text'].tolist())
    _vectorizer.fit([all_docs])

    logger.debug("COHERENCE: vectorizer fitting done")


    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    logger.debug("COHERENCE: extracting features for topic coherence eval done")


    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for _, row in documents_per_topic.iterrows():
        topic_words.append([word for word in row['Documents'].split() if word in dictionary.token2id])

    logger.debug("COHERENCE: extract words in each topic if non empty done")


    topic_words = [words for words in topic_words if len(words) > 0]

    # Evaluate Coherence
    coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence=metric, processes=1)
    coherence_scores = coherence_model.get_coherence_per_topic()

    logger.debug("COHERENCE: evaluation of coherence done")

    return coherence_scores

############################################################################################################

# Sidebar menu for BERTopic hyperparameters
st.sidebar.header("BERTopic Hyperparameters")
language = st.sidebar.selectbox("Select Language", ["English", "French"], key='language')

if language == "English":
    stopwords_list = stopwords.words("english")
    embedding_model_name = st.sidebar.selectbox("Embedding Model", ["all-MiniLM-L12-v2", "all-mpnet-base-v2"], key='embedding_model_name')
elif language == "French" : 
    stopwords_list = stopwords.words("english") + FRENCH_STOPWORDS + STOP_WORDS_RTE
    embedding_model_name = st.sidebar.selectbox("Embedding Model", [ "dangvantuan/sentence-camembert-large", "antoinelouis/biencoder-distilcamembert-mmarcoFR"], key='embedding_model_name')


with st.sidebar.expander("UMAP Hyperparameters", expanded=True):
    umap_n_components = st.number_input("UMAP n_components", value=5, min_value=2, max_value=100, key='umap_n_components')
    umap_n_neighbors = st.number_input("UMAP n_neighbors", value=15, min_value=2, max_value=100, key='umap_n_neighbors')
with st.sidebar.expander("HDBSCAN Hyperparameters", expanded=True):
    hdbscan_min_cluster_size = st.number_input("HDBSCAN min_cluster_size", value=2, min_value=2, max_value=100, key='hdbscan_min_cluster_size')
    hdbscan_min_samples = st.number_input("HDBSCAN min_sample", value=1, min_value=1, max_value=100, key='hdbscan_min_samples')
    hdbscan_cluster_selection_method = st.selectbox("Cluster Selection Method", ["leaf", "eom"], key='hdbscan_cluster_selection_method')
with st.sidebar.expander("Vectorizer Hyperparameters", expanded=True):
    top_n_words = st.number_input("Top N Words", value=10, min_value=1, max_value=50, key='top_n_words')
    vectorizer_ngram_range = st.selectbox("N-Gram range", [(1,2), (1,1), (2,2)], key='vectorizer_ngram_range')
    min_df = st.number_input("min_df", value=1, min_value=1, max_value=50, key='min_df')
with st.sidebar.expander("Merging Hyperparameters", expanded=True):
    min_similarity = st.slider("Minimum Similarity for Merging", 0.0, 1.0, 0.8, 0.01, key='min_similarity')
with st.sidebar.expander("Zero-shot Parameters", expanded=True):
    zeroshot_min_similarity = st.slider("Zeroshot Minimum Similarity", 0.0, 1.0, 0.45, 0.01, key='zeroshot_min_similarity')


cwd_data = cwd+'/data/'
csv_files = glob.glob(os.path.join(cwd_data, '*.csv'))
parquet_files = glob.glob(os.path.join(cwd_data, '*.parquet'))
json_files = glob.glob(os.path.join(cwd_data, '*.json'))
jsonl_files = glob.glob(os.path.join(cwd_data, '*.jsonl'))

file_list = [(os.path.basename(f), os.path.splitext(f)[-1][1:]) for f in csv_files + parquet_files + json_files + jsonl_files]

@st.cache_data
def load_data(selected_file):
    # Get the selected file name and extension
    file_name, file_ext = selected_file

    # Load the data based on the file extension
    if file_ext == 'csv':
        df = pd.read_csv(os.path.join(cwd_data, file_name))
    elif file_ext == 'parquet':
        df = pd.read_parquet(os.path.join(cwd_data, file_name))
    elif file_ext == 'json':
        df = pd.read_json(os.path.join(cwd_data, file_name))
    elif file_ext == 'jsonl':
        df = pd.read_json(os.path.join(cwd_data, file_name), lines=True)

    return df

@st.cache_data
def preprocess_french_data(df):
    df['text'] = df['text'].apply(preprocess_french_text)
    return df

def group_by_days(df, day_granularity=1):
    # Ensure the 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by the specified number of days
    grouped = df.groupby(pd.Grouper(key='timestamp', freq=f'{day_granularity}D'))
    
    # Create a dictionary where each key is the timestamp group and the value is the corresponding dataframe
    dict_of_dfs = {name: group for name, group in grouped}
    
    return dict_of_dfs



selected_file = st.selectbox("Select a dataset", file_list)
st.session_state['raw_df'] = load_data(selected_file)
df = st.session_state['raw_df']

# Preprocess the data if the language is French
if language == "French":
    df = preprocess_french_data(df)

# Add a toggle button to split text by paragraphs
split_by_paragraph = st.checkbox("Split text by paragraphs", value=False, key="split_by_paragraph")

# Split by paragraphs if selected
if split_by_paragraph:
    new_rows = []
    for _, row in df.iterrows():
        paragraphs = row['text'].split('\n\n')
        for paragraph in paragraphs:
            new_row = row.copy()
            new_row['text'] = paragraph
            new_rows.append(new_row)
    df = pd.DataFrame(new_rows)


# Minimum characters input
min_chars = st.number_input("Minimum Characters", value=0, min_value=0, max_value=1000, key='min_chars')
if min_chars > 0: df = df[df['text'].str.len() >= min_chars]

# Remove rows with empty text
df = df[df['text'].str.strip() != ''].reset_index(drop=True)

# Select timeframe
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
start_date, end_date = st.slider("Select Timeframe", min_value=min_date, max_value=max_date, value=(min_date, max_date), key='timeframe_slider')

# Filter the DataFrame based on the selected timeframe
df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].sort_values(by='timestamp').reset_index(drop=True)



st.session_state['timefiltered_df'] = df_filtered
st.write(f"Number of documents in selected timeframe: {len(st.session_state['timefiltered_df'])}")

# Zero-shot topic definition
zeroshot_topic_list = st.text_input("Enter zero-shot topics (separated by /)", value="")
zeroshot_topic_list = [topic.strip() for topic in zeroshot_topic_list.split("/")]

# Embed documents
if st.button("Embed Documents"):
    with st.spinner("Embedding documents..."):
        st.session_state.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
        st.session_state.embedding_model.max_seq_length = 512
        embeddings = st.session_state.embedding_model.encode(st.session_state['timefiltered_df']['text'].tolist(), show_progress_bar=True)
        st.session_state.embeddings = embeddings
    st.success("Embeddings calculated successfully!")
else:
    st.session_state.embeddings = st.session_state.get('embeddings', None)

# Train models button
if 'timefiltered_df' in st.session_state and len(st.session_state.timefiltered_df) > 0:

    # Convert 'text' column to strings
    st.session_state['timefiltered_df']['text'] = st.session_state['timefiltered_df']['text'].astype(str)

    # Select granularity
    granularity = st.number_input("Select Granularity", value=7, min_value=1, max_value=30, key='granularity_select')

    # Show documents per grouped timestamp
    with st.expander("Documents per Timestamp", expanded=True):
        
        grouped_data = group_by_days(st.session_state['timefiltered_df'], day_granularity=granularity)

        non_empty_timestamps = [timestamp for timestamp, group in grouped_data.items() if not group.empty]
        if len(non_empty_timestamps) > 0:
            selected_timestamp = st.select_slider("Select Timestamp", options=non_empty_timestamps, key='timestamp_slider')
            selected_docs = grouped_data[selected_timestamp]
            st.dataframe(selected_docs[['timestamp', 'text']], use_container_width=True)
        else:
            st.warning("No data available for the selected granularity.")

    # Train Models
    if st.button("Train Models"):
        # Set up progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Set up BERTopic components
        umap_model = UMAP(n_components=umap_n_components, n_neighbors=umap_n_neighbors, random_state=42, metric="cosine", verbose=False)
        hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric='euclidean', cluster_selection_method=hdbscan_cluster_selection_method, prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words=stopwords_list, min_df=min_df, ngram_range=vectorizer_ngram_range)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        if zeroshot_topic_list == [""]:
            logger.warning("Warning: No zeroshot topics defined.")
            zeroshot_topic_list= None

        # Create topic models based on selected granularity
        topic_models = {}
        doc_groups = {}
        emb_groups = {}
        non_empty_groups = [(period, group) for period, group in grouped_data.items() if not group.empty]
        for i, (period, group) in enumerate(non_empty_groups):
            docs = group['text'].tolist()
            embeddings_subset = st.session_state.embeddings[group.index]
            try:
                topic_model, docs = create_topic_models(docs, st.session_state.embedding_model, embeddings_subset, umap_model, hdbscan_model, vectorizer_model, mmr_model, top_n_words, zeroshot_topic_list, zeroshot_min_similarity)
                topic_models[period] = topic_model
                doc_groups[period] = docs
                emb_groups[period] = embeddings_subset
            except:
                logger.debug(f"There isn't enough data in the dataframe corresponding to the period {period}. Skipping...")
                continue

            # Update progress bar
            progress = (i + 1) / len(non_empty_groups)
            progress_bar.progress(progress)
            progress_text.text(f"Training BERTopic model for {period} PARAMS USED : UMAP n_components: {umap_n_components}, UMAP n_neighbors: {umap_n_neighbors}, HDBSCAN min_cluster_size: {hdbscan_min_cluster_size}, HDBSCAN min_samples: {hdbscan_min_samples}, HDBSCAN cluster_selection_method: {hdbscan_cluster_selection_method}, Vectorizer min_df: {min_df}, MMR diversity: 0.3, Top N Words: {top_n_words}, Zeroshot Topics: {zeroshot_topic_list}, Zero-shot Min Similarity: {zeroshot_min_similarity}, Embedding Model: {embedding_model_name}")

        # Save topic models and doc_groups to session state
        st.session_state.topic_models = topic_models
        st.session_state.doc_groups = doc_groups
        st.session_state.emb_groups = emb_groups

        # Notify when training is complete
        st.success("Model training complete!")

    



# Display Results
if 'topic_models' in st.session_state:
    topic_models = st.session_state.topic_models
    model_periods = list(topic_models.keys())

    # Plot number of topics detected for each model
    num_topics = [len(model.get_topic_info()) for model in topic_models.values()]
    fig_num_topics = go.Figure(data=[go.Bar(x=list(topic_models.keys()), y=num_topics)])
    fig_num_topics.update_layout(title="Number of Topics Detected", xaxis_title="Time Period", yaxis_title="Number of Topics")
    st.plotly_chart(fig_num_topics, use_container_width=True)

    # Plot size of outlier topic for each model
    outlier_sizes = [model.get_topic_info().loc[model.get_topic_info()['Topic'] == -1]['Count'].values[0] if -1 in model.get_topic_info()['Topic'].values else 0 for model in topic_models.values()]
    fig_outlier_sizes = go.Figure(data=[go.Bar(x=list(topic_models.keys()), y=outlier_sizes)])
    fig_outlier_sizes.update_layout(title="Size of Outlier Topic", xaxis_title="Time Period", yaxis_title="Size of Outlier Topic")
    st.plotly_chart(fig_outlier_sizes, use_container_width=True)
        
    with st.expander("Topic per timestamp", expanded=True):
        selected_model_period = st.select_slider("Select Model", options=model_periods, key='model_slider')
        selected_model = topic_models[selected_model_period]
        selected_docs = st.session_state.doc_groups[selected_model_period]
        st.dataframe(selected_model.get_document_info(docs=selected_docs), use_container_width=True)    
        st.dataframe(selected_model.get_topic_info(), use_container_width=True)


    # Display weak signal trend
    if zeroshot_topic_list is not None:
        weak_signal_trends = detect_weak_signals_zeroshot(topic_models, zeroshot_topic_list)

    if zeroshot_topic_list is not None and not all(not value for value in weak_signal_trends.values()):
        with st.expander("Zero-shot Weak Signal Trends", expanded=True):
            # Define the Max Popularity and History values
            max_popularity = st.number_input("Max Popularity", min_value=0, max_value=1000, value=5)
            history = st.number_input("History (in days)", min_value=1, max_value=100, value=7)

            # Create a single figure for all topics
            fig_trend = go.Figure()

            for topic, weak_signal_trend in weak_signal_trends.items():
                cumulative_count = []
                hovertext = []
                last_update_timestamp = None

                for timestamp in sorted(weak_signal_trend.keys()):
                    if last_update_timestamp is None:
                        last_update_timestamp = timestamp

                    days_since_last_update = (timestamp - last_update_timestamp).days

                    # Apply degradation if no update on the current timestamp
                    if days_since_last_update > 0:
                        degradation_factor = days_since_last_update / history
                        cumulative_count[-1] -= cumulative_count[-1] * degradation_factor

                        # Reset the count to 0 if it falls below 0 due to degradation
                        if cumulative_count[-1] < 0:
                            cumulative_count[-1] = 0

                    # Get the count for the current timestamp
                    count = weak_signal_trend[timestamp]['Count']

                    # Add the count to the degraded cumulative count
                    cumulative_count.append(cumulative_count[-1] + count if cumulative_count else count)
                    last_update_timestamp = timestamp

                    # Prepare the hover text for the current timestamp
                    representation = weak_signal_trend[timestamp]['Representation']
                    hovertext.append(f"Topic: {topic}<br>Representation: {representation}<br>Count: {count}")

                # Create a scatter plot trace for each topic with the cumulative count and hover text
                fig_trend.add_trace(go.Scatter(x=sorted(weak_signal_trend.keys()), y=cumulative_count, mode='lines+markers', name=topic, hovertext=hovertext, hoverinfo='text'))

            # Collect all timestamps from weak_signal_trends
            all_timestamps = set()
            for weak_signal_trend in weak_signal_trends.values():
                all_timestamps.update(weak_signal_trend.keys())

            # Find the oldest and newest timestamps
            oldest_timestamp = min(all_timestamps)
            newest_timestamp = max(all_timestamps)

            # Add a horizontal line for the Max Popularity threshold
            fig_trend.add_shape(type="line", x0=oldest_timestamp, y0=max_popularity, x1=newest_timestamp, y1=max_popularity, line=dict(color="green", width=2, dash="dash"))

            # Add a horizontal line at 0 to indicate the noise level
            fig_trend.add_shape(type="line", x0=oldest_timestamp, y0=0, x1=newest_timestamp, y1=0, line=dict(color="red", width=2, dash="dash"))
            
            # Update the plot layout with title and axis labels
            fig_trend.update_layout(title="Cumulative Frequency of Weak Signals with Degradation", xaxis_title="Timestamp", yaxis_title="Cumulative Frequency")

            # Display the plot using Streamlit
            st.plotly_chart(fig_trend, use_container_width=True)

            # Display topic details in a DataFrame for each topic
            for topic, weak_signal_trend in weak_signal_trends.items():
                st.subheader(f"Topic Details for Topic: {topic}")
                
                topic_details = []
                for timestamp, data in weak_signal_trend.items():
                    topic_details.append({
                        'Representation': data['Representation'],
                        'Count': data['Count'],
                        'Representative_Docs': ', '.join(data['Representative_Docs']),
                        'Timestamp': timestamp
                    })

                topic_details_df = pd.DataFrame(topic_details)
                st.dataframe(topic_details_df, use_container_width=True)

    # Merge models button
    if st.button("Merge Models"):
        with st.spinner("Merging models..."):

            # Preprocess topic models
            topic_dfs = {}
            for period, topic_model in st.session_state.topic_models.items():
                docs = st.session_state.doc_groups[period]
                embeddings = st.session_state.emb_groups[period]
                topic_dfs[period] = preprocess_model(topic_model, docs, embeddings)

            # Merge models
            timestamps = sorted(topic_dfs.keys())  # Sort timestamps in ascending order
            merged_df_without_outliers = None
            all_merge_histories = []  # Store all merge histories
            all_new_topics = []  # Store all newly added topics
            
            progress_bar = st.progress(0)
            
            for i in range(len(timestamps) - 1):

                current_timestamp = timestamps[i]
                next_timestamp = timestamps[i+1]

                df1 = topic_dfs[current_timestamp]
                df1 = df1[df1['Topic'] != -1]

                df2 = topic_dfs[next_timestamp]
                df2 = df2[df2['Topic'] != -1]

                if merged_df_without_outliers is None:
                    merged_df_without_outliers, merge_history, new_topics = merge_models(df1, 
                                                                                         df2, 
                                                                                         min_similarity=min_similarity, 
                                                                                         timestamp=current_timestamp)
                    
                    st.write("First merging process")

                else:
                    merged_df_without_outliers, merge_history, new_topics = merge_models(merged_df_without_outliers, 
                                                                                         df2, 
                                                                                         min_similarity=min_similarity, 
                                                                                         timestamp=current_timestamp)

                                
                all_merge_histories.append(merge_history)  # Store the merge history at each timestep
                all_new_topics.append(new_topics)  # Store the newly added topics at each timestep
                
                # Update progress bar
                progress = (i + 1) / len(timestamps)
                progress_bar.progress(progress)
            
            # Concatenate all merge histories into a single dataframe
            all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
            
            # Concatenate all newly added topics into a single dataframe
            all_new_topics_df = pd.concat(all_new_topics, ignore_index=True)
            
            # Save merged_df, all_merge_histories_df, and all_new_topics_df to session state
            st.session_state.merged_df = merged_df_without_outliers
            st.session_state.all_merge_histories_df = all_merge_histories_df
            st.session_state.all_new_topics_df = all_new_topics_df

        st.success("Model merging complete!")
        
    st.write("Merging process")
    if "all_merge_histories_df" in st.session_state and not st.session_state.all_merge_histories_df.empty:
        # Display topic evolution plot with history and max popularity
        with st.expander("Topic Size Evolution", expanded=True):
            fig = go.Figure()

            # Create a dictionary to store the cumulative sum of topic sizes
            topic_sizes = {}

            # Define the Max Popularity and History values
            max_popularity = st.number_input("Max Popularity", min_value=0, max_value=1000, value=5, key='topic_size_max_popularity')
            history = st.number_input("History (in days)", min_value=1, max_value=100, value=7, key='topic_size_history')

            # Iterate over each row in the all_merge_histories_df dataframe
            for _, row in st.session_state.all_merge_histories_df.iterrows():
                topic1 = row['Topic1']
                timestamp = row['Timestamp']
                count1 = row['Count1']
                
                if topic1 not in topic_sizes:
                    topic_sizes[topic1] = {'Timestamp': [timestamp], 'Popularity': [count1]}
                else:
                    last_update_timestamp = topic_sizes[topic1]['Timestamp'][-1]
                    days_since_last_update = (timestamp - last_update_timestamp).days

                    # Apply degradation if no update on the current timestamp
                    if days_since_last_update > 0:
                        degradation_factor = days_since_last_update / history
                        topic_sizes[topic1]['Popularity'][-1] -= topic_sizes[topic1]['Popularity'][-1] * degradation_factor

                        # Reset the size to 0 if it falls below 0 due to degradation
                        if topic_sizes[topic1]['Popularity'][-1] < 0:
                            topic_sizes[topic1]['Popularity'][-1] = 0

                    topic_sizes[topic1]['Timestamp'].append(timestamp)
                    topic_sizes[topic1]['Popularity'].append(topic_sizes[topic1]['Popularity'][-1] + count1)
            
            # Add traces for each topic
            for topic, data in topic_sizes.items():
                fig.add_trace(go.Scatter(
                    x=data['Timestamp'],
                    y=data['Popularity'],
                    mode='lines+markers',
                    name=f'Topic {topic}',
                    hovertemplate='Topic: %{text}<br>Popularity: %{y}<br>Timestamp: %{x}<extra></extra>',
                    text=[st.session_state.all_merge_histories_df[(st.session_state.all_merge_histories_df['Topic1'] == topic) & (st.session_state.all_merge_histories_df['Timestamp'] == ts)]['Representation1'].values[0] if i == 0 else st.session_state.all_merge_histories_df[(st.session_state.all_merge_histories_df['Topic1'] == topic) & (st.session_state.all_merge_histories_df['Timestamp'] == ts)]['Representation2'].values[0] for i, ts in enumerate(data['Timestamp'])]
                ))
            
            # Find the oldest and newest timestamps
            all_timestamps = [timestamp for topic_data in topic_sizes.values() for timestamp in topic_data['Timestamp']]
            oldest_timestamp = min(all_timestamps)
            newest_timestamp = max(all_timestamps)

            # Add a horizontal line for the Max Popularity threshold
            fig.add_shape(type="line", x0=oldest_timestamp, y0=max_popularity, x1=newest_timestamp, y1=max_popularity, line=dict(color="green", width=2, dash="dash"))

            # Add a horizontal line at 0 to indicate the noise level
            fig.add_shape(type="line", x0=oldest_timestamp, y0=0, x1=newest_timestamp, y1=0, line=dict(color="red", width=2, dash="dash"))
            
            # Update the layout
            fig.update_layout(
                title='Topic Size Evolution',
                xaxis_title='Timestamp',
                yaxis_title='Topic Size',
                hovermode='x',
                legend_title='Topics'
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        
        # Call the transform_dataframe function with your dataframe
        transformed_df = transform_dataframe(st.session_state.all_merge_histories_df)
        # Create search box and slider using Streamlit
        search_term = st.text_input("Search topics by keyword:")
        max_pairs = st.slider("Max number of topic pairs to display", min_value=1, max_value=1000, value=50)

        # Create the Sankey Diagram
        sankey_diagram = create_sankey_diagram(transformed_df, search_term, max_pairs)

        # Display the diagram using Streamlit in an expander
        with st.expander("Topic Merging Process", expanded=True):
            st.plotly_chart(sankey_diagram, use_container_width=True)

    st.write("Newly detected topics")  
    if "all_new_topics_df" in st.session_state and not st.session_state.all_new_topics_df.empty:
        # New scatter plot for newly emerged topics
        fig_new_topics = go.Figure()

        # Iterate over each timestamp in the all_new_topics_df dataframe
        for timestamp, topics_df in st.session_state.all_new_topics_df.groupby('Timestamp'):
            fig_new_topics.add_trace(go.Scatter(
                x=[timestamp] * len(topics_df),
                y=topics_df['Count'],
                text=topics_df['Topic'],
                mode='markers',
                marker=dict(
                    size=topics_df['Count'],
                    sizemode='area',
                    sizeref=2. * max(topics_df['Count']) / (40. ** 2),
                    sizemin=4
                ),
                hovertemplate=(
                    'Timestamp: %{x}<br>'
                    'Topic ID: %{text}<br>'
                    'Count: %{y}<br>'
                    'Representation: %{customdata}<extra></extra>'  # Adding the representation to hover text
                ),
                customdata=topics_df['Representation']  # Supplying the representation data
            ))

        # Update the layout
        fig_new_topics.update_layout(
            title='Newly Emerged Topics',
            xaxis_title='Timestamp',
            yaxis_title='Topic Size',
            showlegend=False
        )

        # New scatter plot for newly emerged topics in an expander
        with st.expander("Newly Emerged Topics", expanded=True):
            st.dataframe(st.session_state.all_new_topics_df[['Topic', 'Count', 'Representation', 'Documents', 'Timestamp']].sort_values(by=['Timestamp', 'Count'], ascending=[True, False]))
            st.plotly_chart(fig_new_topics, use_container_width=True)
            
        # Coherence calculation parameters
        metric = st.selectbox("Coherence Metric", ["c_v", "c_npmi", "u_mass"])
        num_signals = st.number_input("Number of Signals", value=5, min_value=1)
        num_docs = st.number_input("Number of Documents per Signal", value=3, min_value=1)

        if st.button("Detect"):
            if "all_new_topics_df" in st.session_state:
                new_topics_df = st.session_state.all_new_topics_df
                vectorizer = CountVectorizer(stop_words=stopwords_list, min_df=min_df, ngram_range=vectorizer_ngram_range)


                with st.spinner("Calculating coherence of topics...") : 
                    st.session_state.top_topics, st.session_state.bottom_topics = detect_weak_signals(new_topics_df, metric, num_signals, num_docs, vectorizer)

        if 'top_topics' in st.session_state and 'bottom_topics' in st.session_state:
            # Display the top topics (coherent signals)
            with st.expander("Coherent Signals"):
                if len(st.session_state.top_topics) < num_signals:
                    logger.warning(f"Only {len(st.session_state.top_topics)} coherent signals found.")

                for _, topic in st.session_state.top_topics.iterrows():
                    st.subheader(f"Topic: {topic['Topic']}")
                    st.write(f"Representation: {topic['Representation']}")
                    st.write(f"Timestamp: {topic['Timestamp']}")
                    st.write(f"Coherence Score: {topic['Coherence']:.3f}")
                    st.write(f"Count: {topic['Count']}")
                    st.write(f"Coherence-Count Score: {topic['Coherence_Count_Score']:.3f}")

                    docs = topic['Documents'][:num_docs]
                    for i, doc in enumerate(docs, start=1):
                        st.write(f"Document {i}:")
                        st.write(doc)
                        st.write("---")

            # Display the bottom topics (noise)
            with st.expander("Noise"):
                if len(st.session_state.bottom_topics) < num_signals:
                    logger.warning(f"Only {len(st.session_state.bottom_topics)} noise topics found.")

                for _, topic in st.session_state.bottom_topics.iterrows():
                    st.subheader(f"Topic: {topic['Topic']}")
                    st.write(f"Representation: {topic['Representation']}")
                    st.write(f"Timestamp: {topic['Timestamp']}")
                    st.write(f"Coherence Score: {topic['Coherence']:.3f}")
                    st.write(f"Count: {topic['Count']}")
                    st.write(f"Coherence-Count Score: {topic['Coherence_Count_Score']:.3f}")

                    docs = topic['Documents'][:num_docs]
                    for i, doc in enumerate(docs, start=1):
                        st.write(f"Document {i}:")
                        st.write(doc)
                        st.write("---")

