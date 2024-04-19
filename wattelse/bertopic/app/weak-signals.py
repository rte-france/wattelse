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
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, docs

# Function to merge BERTopic models
def merge_topic_models(topic_models, min_similarity, zeroshot_topic_list):
    merged_model = None
    weak_signal_trend = {}
    for i, (period, topic_model) in enumerate(tqdm(topic_models.items(), desc="Merging models", unit="model")):
        if merged_model is None:
            merged_model = topic_model
            topic_info = merged_model.get_topic_info()
            for topic_id, topic_name in zip(topic_info['Topic'], topic_info['Name']):
                if topic_name in zeroshot_topic_list:
                    weak_signal_trend[topic_name] = {period: merged_model.get_topic_freq(topic=topic_id)}
        else:
            prev_model = merged_model
            merged_model = BERTopic.merge_models([merged_model, topic_model], min_similarity=min_similarity)
            topic_info = merged_model.get_topic_info()
            for topic_id, topic_name in zip(topic_info['Topic'], topic_info['Name']):
                if topic_name in zeroshot_topic_list:
                    if topic_name not in weak_signal_trend:
                        weak_signal_trend[topic_name] = {period: merged_model.get_topic_freq(topic=topic_id)}
                    else:
                        weak_signal_trend[topic_name][period] = merged_model.get_topic_freq(topic=topic_id)
    return merged_model, weak_signal_trend



# Function to look out for weak signals based on the zero-shot list of topics we want to monitor
def detect_weak_signals(topic_models, zeroshot_topic_list):
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


# Function to preprocess topic models
def preprocess_model(topic_model, docs):
    # Get topic information
    topic_info = topic_model.get_topic_info()
    # Get document information
    doc_info = topic_model.get_document_info(docs)
    # Group documents by topic
    doc_groups = doc_info.groupby('Topic')['Document'].apply(list)
    # Create a DataFrame with topic information and document groups
    topic_df = pd.DataFrame({
        'Topic': topic_info['Topic'],
        'Count': topic_info['Count'],
        'Representation': topic_info['Representation'],
        'Documents': doc_groups.reindex(topic_info['Topic']).tolist(),
        'Embedding': topic_model.topic_embeddings_.tolist()
    })
    return topic_df







def merge_models(df1, df2, min_similarity, timestamp):
    merged_df = df1.copy()
    merge_history = []

    for _, row2 in df2.iterrows():
        topic2 = row2['Topic']
        count2 = row2['Count']
        representation2 = row2['Representation']
        documents2 = row2['Documents']
        embedding2 = row2['Embedding']

        max_similarity = -1
        max_similar_topic = None

        for _, row1 in df1.iterrows():
            embedding1 = row1['Embedding']
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                max_similar_topic = row1['Topic']

        if max_similarity < min_similarity:
            # Add a new entry to the merged dataframe
            new_topic = merged_df['Topic'].max() + 1 if not merged_df.empty else 0
            new_entry = pd.DataFrame({
                'Topic': [new_topic],
                'Count': [count2],
                'Representation': [representation2],
                'Documents': [documents2],
                'Embedding': [embedding2]
            })
            merged_df = pd.concat([merged_df, new_entry], ignore_index=True)
        else:
            # Merge with the most similar topic from df1
            similar_row = merged_df[merged_df['Topic'] == max_similar_topic].iloc[0]
            count1 = similar_row['Count']
            documents1 = similar_row['Documents']
            embedding1 = similar_row['Embedding']

            merged_count = count1 + count2
            merged_documents = documents1 + documents2

            weight1 = count1 / (count1 + count2)
            weight2 = count2 / (count1 + count2)
            merged_embedding = weight1 * np.array(embedding1) + weight2 * np.array(embedding2)

            index = merged_df[merged_df['Topic'] == max_similar_topic].index[0]
            merged_df.at[index, 'Count'] = merged_count
            merged_df.at[index, 'Documents'] = merged_documents
            merged_df.at[index, 'Embedding'] = merged_embedding

            # Log the merge history
            merge_history.append({
                'Timestamp': timestamp,
                'Topic1': max_similar_topic,
                'Topic2': topic2,
                'Representation1': similar_row['Representation'],
                'Representation2': representation2,
                'Embedding1': embedding1,
                'Embedding2': embedding2,
                'Similarity': max_similarity,
                'Count1': count1,
                'Count2': count2
            })

    return merged_df, pd.DataFrame(merge_history)


def transform_dataframe(df):
    # Create a copy of the original dataframe
    transformed_df = df.copy()
    
    # Create a column 'Timestamp_Index' that maps each timestamp to an index
    timestamps = transformed_df['Timestamp'].unique()
    timestamp_index_map = {timestamp: index for index, timestamp in enumerate(timestamps)}
    transformed_df['Timestamp_Index'] = transformed_df['Timestamp'].map(timestamp_index_map)
    
    # Group by Topic1 and collect the list of timestamp indices where each Topic1 value appears
    topic1_timestamp_indices = transformed_df.groupby('Topic1')['Timestamp_Index'].apply(list).to_dict()
    
    logger.debug(f"Timestamp Index Map: {timestamp_index_map}")
    logger.debug(f"Topic1 Timestamp Indices: {topic1_timestamp_indices}")
    
    # Initialize variables to store the source, destination, representation, timestamp, and count values
    src_values = []
    dest_values = []
    representation_values = []
    timestamp_values = []
    count_values = []
    
    # Initialize a dictionary to store the mapping of (topic1, timestamp_index) to the new destination topic
    topic1_dest_map = {}
    
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
            
            # Append the source, destination, representation, timestamp, and count values to the respective lists
            src_values.extend([src_topic1, src_topic2])
            dest_values.extend([dest_topic, dest_topic])
            representation_values.extend([representation1, representation2])
            timestamp_values.extend([timestamp, timestamp])
            count_values.extend([count1, count2])
    
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
    
    # Drop duplicate rows based on all columns
    transformed_df_new = transformed_df_new.drop_duplicates()
    
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
                          'Destination Topic ID: %{customdata[2]}<br>' +
                          'Representation: %{customdata[3]}<br>' +
                          'Count: %{customdata[4]}<extra></extra>',
            customdata=[[
                link['timestamp'],
                link['source_topic_id'],
                link['target_topic_id'],
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


############################################################################################################

# Sidebar menu for BERTopic hyperparameters
st.sidebar.header("BERTopic Hyperparameters")
embedding_model_name = st.sidebar.selectbox("Embedding Model", ["antoinelouis/biencoder-distilcamembert-mmarcoFR", "dangvantuan/sentence-camembert-large", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "thenlper/gte-small"], key='embedding_model_name')
with st.sidebar.expander("UMAP Hyperparameters", expanded=True):
    umap_n_components = st.number_input("UMAP n_components", value=5, min_value=2, max_value=100, key='umap_n_components')
    umap_n_neighbors = st.number_input("UMAP n_neighbors", value=15, min_value=2, max_value=100, key='umap_n_neighbors')
with st.sidebar.expander("HDBSCAN Hyperparameters", expanded=True):
    hdbscan_min_cluster_size = st.number_input("HDBSCAN min_cluster_size", value=2, min_value=2, max_value=100, key='hdbscan_min_cluster_size')
    hdbscan_min_samples = st.number_input("HDBSCAN min_sample", value=1, min_value=1, max_value=100, key='hdbscan_min_samples')
    hdbscan_cluster_selection_method = st.selectbox("Cluster Selection Method", ["eom", "leaf"], key='hdbscan_cluster_selection_method')
with st.sidebar.expander("Vectorizer Hyperparameters", expanded=True):
    top_n_words = st.number_input("Top N Words", value=10, min_value=1, max_value=50, key='top_n_words')
    min_df = st.number_input("min_df", value=1, min_value=1, max_value=50, key='min_df')
with st.sidebar.expander("Merging Hyperparameters", expanded=True):
    min_similarity = st.slider("Minimum Similarity for Merging", 0.0, 1.0, 0.9, 0.01, key='min_similarity')
with st.sidebar.expander("Zero-shot Parameters", expanded=True):
    zeroshot_min_similarity = st.slider("Zeroshot Minimum Similarity", 0.0, 1.0, 0.45, 0.01, key='zeroshot_min_similarity')

# Load data
cwd = os.getcwd()
csv_files = glob.glob(os.path.join(cwd, '*.csv'))
parquet_files = glob.glob(os.path.join(cwd, '*.parquet'))
json_files = glob.glob(os.path.join(cwd, '*.json'))
jsonl_files = glob.glob(os.path.join(cwd, '*.jsonl'))

file_list = [(os.path.basename(f), os.path.splitext(f)[-1][1:]) for f in csv_files + parquet_files + json_files + jsonl_files]

@st.cache_data
def load_data(selected_file):
    # Get the selected file name and extension
    file_name, file_ext = selected_file

    # Load the data based on the file extension
    if file_ext == 'csv':
        df = pd.read_csv(os.path.join(cwd, file_name))
    elif file_ext == 'parquet':
        df = pd.read_parquet(os.path.join(cwd, file_name))
    elif file_ext == 'json':
        df = pd.read_json(os.path.join(cwd, file_name))
    elif file_ext == 'jsonl':
        df = pd.read_json(os.path.join(cwd, file_name), lines=True)

    return df

selected_file = st.selectbox("Select a dataset", file_list)
st.session_state['raw_df'] = load_data(selected_file)
df = st.session_state['raw_df']

# Add a toggle button to split text by paragraphs
split_by_paragraph = st.checkbox("Split text by paragraphs", value=False, key="split_by_paragraph")

# Select timeframe
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
start_date, end_date = st.slider("Select Timeframe", min_value=min_date, max_value=max_date, value=(min_date, max_date), key='timeframe_slider')

# Filter the DataFrame based on the selected timeframe
df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

if split_by_paragraph:
    # Split the texts in the "text" column by paragraphs and create new rows for each paragraph
    new_rows = []
    for _, row in df_filtered.iterrows():
        paragraphs = row['text'].split('\n\n')
        for paragraph in paragraphs:
            new_row = row.copy()
            new_row['text'] = paragraph
            new_rows.append(new_row)
    df_filtered = pd.DataFrame(new_rows)

# remove rows with empty text
df_filtered = df_filtered[df_filtered['text'].str.strip() != '']

# Reset the index of the filtered DataFrame
df_filtered.reset_index(drop=True, inplace=True)


st.session_state['timefiltered_df'] = df_filtered
st.write(f"Number of documents in selected timeframe: {len(st.session_state['timefiltered_df'])}")

# Zero-shot topic definition
zeroshot_topic_list = st.text_input("Enter zero-shot topics (separated by /)", value="Viruses, diseases, pandemics outbreaks, WHO, Health.")
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
    # Select granularity
    granularity = st.selectbox("Select Granularity", ["Day", "Week", "Month"], key='granularity_selectbox')

    # Show documents per grouped timestamp
    with st.expander("Documents per Timestamp", expanded=True):
        if granularity == "Day":
            grouped_data = st.session_state['timefiltered_df'].groupby(st.session_state['timefiltered_df']['timestamp'].dt.date)
        elif granularity == "Week":
            grouped_data = st.session_state['timefiltered_df'].groupby(pd.Grouper(key='timestamp', freq='W'))
        else:  # Month
            grouped_data = st.session_state['timefiltered_df'].groupby(pd.Grouper(key='timestamp', freq='M'))

        timestamps = list(grouped_data.groups.keys())
        selected_timestamp = st.select_slider("Select Timestamp", options=timestamps, key='timestamp_slider')

        selected_docs = grouped_data.get_group(selected_timestamp)
        st.dataframe(selected_docs[['timestamp', 'text']], use_container_width=True)
        
    if st.button("Train Models"):
        # Set up progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Create topic models based on selected granularity
        topic_models = {}
        doc_groups = {}
        for i, (period, group) in enumerate(grouped_data):
            docs = group['text'].tolist()
            embeddings_subset = st.session_state.embeddings[group.index]
            umap_model = UMAP(n_components=umap_n_components, n_neighbors=umap_n_neighbors, random_state=42, metric="cosine", verbose=False)
            hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric='euclidean', cluster_selection_method=hdbscan_cluster_selection_method, prediction_data=True)
            vectorizer_model = CountVectorizer(stop_words=stopwords.words('english'), min_df=min_df, ngram_range=(1, 2))
            mmr_model = MaximalMarginalRelevance(diversity=0.3)

            topic_model, docs = create_topic_models(docs, st.session_state.embedding_model, embeddings_subset, umap_model, hdbscan_model, vectorizer_model, mmr_model, top_n_words, zeroshot_topic_list, zeroshot_min_similarity)
            topic_models[period] = topic_model
            doc_groups[period] = docs

            # Update progress bar
            progress = (i + 1) / len(grouped_data)
            progress_bar.progress(progress)
            progress_text.text(f"Training BERTopic model for {period} PARAMS USED : UMAP n_components: {umap_n_components}, UMAP n_neighbors: {umap_n_neighbors}, HDBSCAN min_cluster_size: {hdbscan_min_cluster_size}, HDBSCAN min_samples: {hdbscan_min_samples}, HDBSCAN cluster_selection_method: {hdbscan_cluster_selection_method}, Vectorizer min_df: {min_df}, MMR diversity: 0.3, Top N Words: {top_n_words}, Zeroshot Topics: {zeroshot_topic_list}, Zero-shot Min Similarity: {zeroshot_min_similarity}, Embedding Model: {embedding_model_name}")

        # Save topic models and doc_groups to session state
        st.session_state.topic_models = topic_models
        st.session_state.doc_groups = doc_groups

        # Notify when training is complete
        st.success("Model training complete!")
        
        # Detect weak signals
        weak_signal_trend = detect_weak_signals(topic_models, zeroshot_topic_list)
        st.session_state.weak_signal_trend = weak_signal_trend
    



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
    weak_signal_trends = detect_weak_signals(topic_models, zeroshot_topic_list)


    with st.expander("Zero-shot Weak Signal Trends", expanded=True):
        # Define the Max Popularity and History values
        max_popularity = st.number_input("Max Popularity", min_value=0, max_value=1000, value=50)
        history = st.number_input("History (in days)", min_value=1, max_value=100, value=1)

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

        # Add a horizontal line for the Max Popularity threshold
        fig_trend.add_shape(type="line", x0=min(weak_signal_trend.keys()), y0=max_popularity, x1=max(weak_signal_trend.keys()), y1=max_popularity, line=dict(color="green", width=2, dash="dash"))

        # Add a horizontal line at 0 to indicate the noise level
        fig_trend.add_shape(type="line", x0=min(weak_signal_trend.keys()), y0=0, x1=max(weak_signal_trend.keys()), y1=0, line=dict(color="red", width=2, dash="dash"))

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
            for period in st.session_state.topic_models.keys():
                docs = st.session_state.doc_groups[period]
                topic_model = st.session_state.topic_models[period]
                logger.debug(f"{len(docs)}, {topic_model.get_topic_info()['Count'].sum()}")

                topic_dfs[period] = preprocess_model(topic_model, docs)
            
            # Save topic_dfs and doc_groups to session state
            st.session_state.topic_dfs = topic_dfs
            st.session_state.doc_groups = st.session_state.doc_groups
            
            # Merge models
            timestamps = sorted(topic_dfs.keys())  # Sort timestamps in ascending order
            merged_df = topic_dfs[timestamps[0]]  # Initialize merged_df with the first entry
            merged_docs = st.session_state.doc_groups[timestamps[0]]
            merged_dfs = {timestamps[0]: merged_df}  # Store merged dataframes at each timestep
            all_merge_histories = []  # Store all merge histories
            
            progress_bar = st.progress(0)
            
            for i in range(1, len(timestamps)):
                timestamp = timestamps[i]
                topic_df = topic_dfs[timestamp]
                
                # Remove rows with "Topic" equal to -1 (outlier topic) from both dataframes
                merged_df = merged_df[merged_df['Topic'] != -1]
                topic_df = topic_df[topic_df['Topic'] != -1]
                
                prev_df = merged_df.copy()
                merged_df, merge_history = merge_models(merged_df, topic_df, min_similarity=min_similarity, timestamp=timestamp)
                merged_docs.extend(st.session_state.doc_groups[timestamp])
                
                merged_dfs[timestamp] = merged_df  # Store the merged dataframe at each timestep
                all_merge_histories.append(merge_history)  # Store the merge history at each timestep
                
                # Update progress bar
                progress = (i + 1) / len(timestamps)
                progress_bar.progress(progress)
            
            # Concatenate all merge histories into a single dataframe
            all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
            
            # Save merged_df and all_merge_histories_df to session state
            st.session_state.merged_df = merged_df
            st.session_state.all_merge_histories_df = all_merge_histories_df
        
        st.success("Model merging complete!")
        
    # Display merged_df
    if "all_merge_histories_df" in st.session_state:
        st.dataframe(st.session_state.merged_df, use_container_width=True)
        # Display topic evolution plot
        fig = go.Figure()

        # Create a dictionary to store the cumulative sum of topic sizes
        topic_sizes = {}

        # Iterate over each row in the all_merge_histories_df dataframe
        for _, row in st.session_state.all_merge_histories_df.iterrows():
            topic1 = row['Topic1']
            timestamp = row['Timestamp']
            count1 = row['Count1']
            
            if topic1 not in topic_sizes:
                topic_sizes[topic1] = {'Timestamp': [timestamp], 'Size': [count1]}
            else:
                topic_sizes[topic1]['Timestamp'].append(timestamp)
                topic_sizes[topic1]['Size'].append(topic_sizes[topic1]['Size'][-1] + count1)
        
        # Add traces for each topic
        for topic, data in topic_sizes.items():
            fig.add_trace(go.Scatter(
                x=data['Timestamp'],
                y=data['Size'],
                mode='lines+markers',
                name=f'Topic {topic}',
                hovertemplate='Topic: %{text}<br>Size: %{y}<br>Timestamp: %{x}<extra></extra>',
                text=[st.session_state.all_merge_histories_df[(st.session_state.all_merge_histories_df['Topic1'] == topic) & (st.session_state.all_merge_histories_df['Timestamp'] == ts)]['Representation1'].values[0] if i == 0 else st.session_state.all_merge_histories_df[(st.session_state.all_merge_histories_df['Topic1'] == topic) & (st.session_state.all_merge_histories_df['Timestamp'] == ts)]['Representation2'].values[0] for i, ts in enumerate(data['Timestamp'])]
            ))
        
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

        # st.dataframe(st.session_state.all_merge_histories_df, use_container_width=True)
        # st.dataframe(transformed_df, use_container_width=True)

        # Create search box and slider using Streamlit
        search_term = st.text_input("Search topics by keyword:")
        max_pairs = st.slider("Max number of topic pairs to display", min_value=1, max_value=1000, value=50)

        # Create the Sankey Diagram
        sankey_diagram = create_sankey_diagram(transformed_df, search_term, max_pairs)

        # Display the diagram using Streamlit
        st.plotly_chart(sankey_diagram, use_container_width=True)
        
        
        
# st.session_state.all_merge_histories_df = pd.read_json('all_merge_histories_df.json')


