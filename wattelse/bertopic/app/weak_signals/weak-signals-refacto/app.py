import streamlit as st
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import os
import glob
import pickle
import shutil
from loguru import logger
from openai import OpenAI
from bertopic import BERTopic
from prompts import get_prompt

from data_loading import load_and_preprocess_data, group_by_days
from topic_modeling import train_topic_models, merge_models, preprocess_model
from visualizations import plot_num_topics_and_outliers, plot_topics_per_timestamp, plot_topic_size_evolution, plot_newly_emerged_topics, prepare_source_topic_data, create_sankey_diagram
from weak_signals import detect_weak_signals_zeroshot
from global_vars import STOP_WORDS_RTE, COMMON_NGRAMS, FRENCH_STOPWORDS, cwd_data
from nltk.corpus import stopwords

import plotly.graph_objects as go

import numpy as np
from tqdm import tqdm
import os

def save_state():
    cache_dir = "cache"
    
    # Delete the existing cache_dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Recreate an empty cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    state_file = os.path.join(cache_dir, "app_state.pkl")
    embeddings_file = os.path.join(cache_dir, "embeddings.npy")

    state = {
        'selected_file': st.session_state.get('selected_file'),
        'min_chars': st.session_state.get('min_chars'),
        'split_by_paragraph': st.session_state.get('split_by_paragraph'),
        'timeframe_slider': st.session_state.get('timeframe_slider'),
        'language': st.session_state.get('language'),
        'embedding_model_name': st.session_state.get('embedding_model_name'),
        'embedding_model': st.session_state.get('embedding_model'),
        'umap_n_components': st.session_state.get('umap_n_components'),
        'umap_n_neighbors': st.session_state.get('umap_n_neighbors'),
        'hdbscan_min_cluster_size': st.session_state.get('hdbscan_min_cluster_size'),
        'hdbscan_min_samples': st.session_state.get('hdbscan_min_samples'),
        'hdbscan_cluster_selection_method': st.session_state.get('hdbscan_cluster_selection_method'),
        'top_n_words': st.session_state.get('top_n_words'),
        'vectorizer_ngram_range': st.session_state.get('vectorizer_ngram_range'),
        'min_df': st.session_state.get('min_df'),
        'min_similarity': st.session_state.get('min_similarity'),
        'zeroshot_min_similarity': st.session_state.get('zeroshot_min_similarity')
    }

    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

    np.save(embeddings_file, st.session_state.embeddings)

    st.success(f"Application state saved.")

def restore_state():
    cache_dir = "cache"
    state_file = os.path.join(cache_dir, "app_state.pkl")
    embeddings_file = os.path.join(cache_dir, "embeddings.npy")

    if os.path.exists(state_file) and os.path.exists(embeddings_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)

        st.session_state['selected_file'] = state.get('selected_file')
        st.session_state['min_chars'] = state.get('min_chars')
        st.session_state['split_by_paragraph'] = state.get('split_by_paragraph')
        st.session_state['timeframe_slider'] = state.get('timeframe_slider')
        st.session_state['language'] = state.get('language')
        st.session_state['embedding_model_name'] = state.get('embedding_model_name')
        st.session_state['embedding_model'] = state.get('embedding_model')
        st.session_state['umap_n_components'] = state.get('umap_n_components')
        st.session_state['umap_n_neighbors'] = state.get('umap_n_neighbors')
        st.session_state['hdbscan_min_cluster_size'] = state.get('hdbscan_min_cluster_size')
        st.session_state['hdbscan_min_samples'] = state.get('hdbscan_min_samples')
        st.session_state['hdbscan_cluster_selection_method'] = state.get('hdbscan_cluster_selection_method')
        st.session_state['top_n_words'] = state.get('top_n_words')
        st.session_state['vectorizer_ngram_range'] = state.get('vectorizer_ngram_range')
        st.session_state['min_df'] = state.get('min_df')
        st.session_state['min_similarity'] = state.get('min_similarity')
        st.session_state['zeroshot_min_similarity'] = state.get('zeroshot_min_similarity')

        st.session_state.embeddings = np.load(embeddings_file)

        st.success(f"Application state restored.")
    else:
        st.warning(f"No saved state found.")

def save_models():
    cache_dir = "cache"
    models_dir = os.path.join(cache_dir, "models")
    
    # Delete the existing models_dir
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)

    # Recreate an empty models_dir
    os.makedirs(models_dir, exist_ok=True)

    for period, topic_model in st.session_state.topic_models.items():
        model_dir = os.path.join(models_dir, str(period))
        os.makedirs(model_dir, exist_ok=True)
        topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=st.session_state.embedding_model)

        # Save doc_info_df and topic_info_df as separate files
        doc_info_df_file = os.path.join(model_dir, "doc_info_df.pkl")
        topic_info_df_file = os.path.join(model_dir, "topic_info_df.pkl")
        topic_model.doc_info_df.to_pickle(doc_info_df_file)
        topic_model.topic_info_df.to_pickle(topic_info_df_file)

    # Save doc_groups and emb_groups
    doc_groups_file = os.path.join(cache_dir, "doc_groups.pkl")
    emb_groups_file = os.path.join(cache_dir, "emb_groups.pkl")
    with open(doc_groups_file, 'wb') as f:
        pickle.dump(st.session_state.doc_groups, f)
    with open(emb_groups_file, 'wb') as f:
        pickle.dump(st.session_state.emb_groups, f)

    # Save granularity_select value
    granularity_file = os.path.join(cache_dir, "granularity.pkl")
    with open(granularity_file, 'wb') as f:
        pickle.dump(st.session_state.granularity_select, f)

    st.success(f"Models saved.")

def restore_models():
    cache_dir = "cache"
    models_dir = os.path.join(cache_dir, "models")

    if os.path.exists(models_dir):
        topic_models = {}
        for period_dir in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, period_dir)
            if os.path.isdir(model_dir):
                topic_model = BERTopic.load(model_dir)

                # Load doc_info_df and topic_info_df
                doc_info_df_file = os.path.join(model_dir, "doc_info_df.pkl")
                topic_info_df_file = os.path.join(model_dir, "topic_info_df.pkl")
                if os.path.exists(doc_info_df_file) and os.path.exists(topic_info_df_file):
                    topic_model.doc_info_df = pd.read_pickle(doc_info_df_file)
                    topic_model.topic_info_df = pd.read_pickle(topic_info_df_file)
                else:
                    logger.warning(f"doc_info_df or topic_info_df not found for period {period_dir}")

                period = pd.Timestamp(period_dir)
                topic_models[period] = topic_model

        st.session_state.topic_models = topic_models

        # Restore doc_groups and emb_groups
        doc_groups_file = os.path.join(cache_dir, "doc_groups.pkl")
        emb_groups_file = os.path.join(cache_dir, "emb_groups.pkl")
        if os.path.exists(doc_groups_file) and os.path.exists(emb_groups_file):
            with open(doc_groups_file, 'rb') as f:
                st.session_state.doc_groups = pickle.load(f)
            with open(emb_groups_file, 'rb') as f:
                st.session_state.emb_groups = pickle.load(f)
        else:
            logger.warning("doc_groups or emb_groups not found.")

        # Restore granularity_select value
        granularity_file = os.path.join(cache_dir, "granularity.pkl")
        if os.path.exists(granularity_file):
            with open(granularity_file, 'rb') as f:
                st.session_state.granularity_select = pickle.load(f)
        else:
            logger.warning("Granularity value not found.")

        st.success(f"Models restored.")
    else:
        st.warning(f"No saved models found.")

def purge_cache():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success(f"Cache purged.")
    else:
        st.warning(f"No cache found.")

def main():
    st.set_page_config(page_title="BERTopic Topic Detection", layout="wide")

    # Restore Previous Run and Purge Cache buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restore Previous Run"):
            restore_state()
            restore_models()
    with col2:
        if st.button("Purge Cache"):
            purge_cache()

    # Sidebar menu for BERTopic hyperparameters
    st.sidebar.header("BERTopic Hyperparameters")
    language = st.sidebar.selectbox("Select Language", ["English", "French"], key='language')

    if language == "English":
        stopwords_list = stopwords.words("english")
        embedding_model_name = st.sidebar.selectbox("Embedding Model", ["all-MiniLM-L12-v2", "all-mpnet-base-v2"], key='embedding_model_name')
    elif language == "French":
        stopwords_list = stopwords.words("english") + FRENCH_STOPWORDS + STOP_WORDS_RTE + COMMON_NGRAMS
        embedding_model_name = st.sidebar.selectbox("Embedding Model", ["dangvantuan/sentence-camembert-large", "antoinelouis/biencoder-distilcamembert-mmarcoFR"], key='embedding_model_name')

    with st.sidebar.expander("UMAP Hyperparameters", expanded=True):
        umap_n_components = st.number_input("UMAP n_components", value=5, min_value=5, max_value=100, key='umap_n_components')
        umap_n_neighbors = st.number_input("UMAP n_neighbors", value=10, min_value=5, max_value=100, key='umap_n_neighbors')
    with st.sidebar.expander("HDBSCAN Hyperparameters", expanded=True):
        hdbscan_min_cluster_size = st.number_input("HDBSCAN min_cluster_size", value=5, min_value=2, max_value=100, key='hdbscan_min_cluster_size')
        hdbscan_min_samples = st.number_input("HDBSCAN min_sample", value=10, min_value=2, max_value=100, key='hdbscan_min_samples')
        hdbscan_cluster_selection_method = st.selectbox("Cluster Selection Method", ["eom", "leaf"], key='hdbscan_cluster_selection_method')
    with st.sidebar.expander("Vectorizer Hyperparameters", expanded=True):
        top_n_words = st.number_input("Top N Words", value=10, min_value=1, max_value=50, key='top_n_words')
        vectorizer_ngram_range = st.selectbox("N-Gram range", [(1, 2), (1, 1), (2, 2)], key='vectorizer_ngram_range')
        min_df = st.number_input("min_df", value=2, min_value=1, max_value=50, key='min_df')
    with st.sidebar.expander("Merging Hyperparameters", expanded=True):
        min_similarity = st.slider("Minimum Similarity for Merging", 0.0, 1.0, 0.7, 0.01, key='min_similarity')
    with st.sidebar.expander("Zero-shot Parameters", expanded=True):
        zeroshot_min_similarity = st.slider("Zeroshot Minimum Similarity", 0.0, 1.0, 0.4, 0.01, key='zeroshot_min_similarity')




    # Load and preprocess data
    selected_file = st.selectbox("Select a dataset", [(os.path.basename(f), os.path.splitext(f)[-1][1:]) for f in glob.glob(os.path.join(cwd_data, '*'))], key='selected_file')
    min_chars = st.number_input("Minimum Characters", value=100, min_value=0, max_value=1000, key='min_chars')
    split_by_paragraph = st.checkbox("Split text by paragraphs", value=True, key="split_by_paragraph")

    df = load_and_preprocess_data(selected_file, language, min_chars, split_by_paragraph)

    # Select timeframe
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    start_date, end_date = st.slider("Select Timeframe", min_value=min_date, max_value=max_date, value=(min_date, max_date), key='timeframe_slider')



    # Filter the DataFrame based on the selected timeframe and deduplicate the split documents
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].drop_duplicates(subset='text', keep='first')
    df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

    st.session_state['timefiltered_df'] = df_filtered
    st.write(f"Number of documents in selected timeframe: {len(st.session_state['timefiltered_df'])}")

    # Select granularity
    granularity = st.number_input("Select Granularity", value=7, min_value=1, max_value=30, key='granularity_select')

    # Show documents per grouped timestamp
    with st.expander("Documents per Timestamp", expanded=True):
        grouped_data = group_by_days(st.session_state['timefiltered_df'], day_granularity=granularity)
        non_empty_timestamps = [timestamp for timestamp, group in grouped_data.items() if not group.empty]
        if len(non_empty_timestamps) > 0:
            selected_timestamp = st.select_slider("Select Timestamp", options=non_empty_timestamps, key='timestamp_slider')
            selected_docs = grouped_data[selected_timestamp]
            st.dataframe(selected_docs[['timestamp', 'text', 'document_id', 'source', 'url']], use_container_width=True)
        else:
            st.warning("No data available for the selected granularity.")

    # Zero-shot topic definition
    zeroshot_topic_list = st.text_input("Enter zero-shot topics (separated by /)", value="")
    zeroshot_topic_list = [topic.strip() for topic in zeroshot_topic_list.split("/")]

    # Embed documents
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            st.session_state.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
            st.session_state.embedding_model.max_seq_length = 512

            texts = st.session_state['timefiltered_df']['text'].tolist()
            batch_size = 1000
            num_batches = (len(texts) + batch_size - 1) // batch_size

            embeddings = []
            for i in tqdm(range(num_batches), desc="Batches processed"):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                batch_embeddings = st.session_state.embedding_model.encode(batch_texts, show_progress_bar=False)
                embeddings.append(batch_embeddings)

            embeddings = np.concatenate(embeddings, axis=0)
            st.session_state.embeddings = embeddings

        st.success("Embeddings calculated successfully!")
        save_state()
    else:
        st.session_state.embeddings = st.session_state.get('embeddings', None)

    # Train models button
    if 'timefiltered_df' in st.session_state and len(st.session_state.timefiltered_df) > 0:

        # Convert 'text' column to strings
        st.session_state['timefiltered_df']['text'] = st.session_state['timefiltered_df']['text'].astype(str)

        # Set up BERTopic components
        umap_model = UMAP(n_components=umap_n_components, n_neighbors=umap_n_neighbors, random_state=42, metric="cosine")
        hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric='euclidean',
                                cluster_selection_method=hdbscan_cluster_selection_method, prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words=stopwords_list, min_df=min_df, ngram_range=vectorizer_ngram_range)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # Train Models
        if st.button("Train Models"):
            topic_models, doc_groups, emb_groups = train_topic_models(
                grouped_data, st.session_state.embedding_model, st.session_state.embeddings,
                umap_model, hdbscan_model, vectorizer_model, mmr_model,
                top_n_words, zeroshot_topic_list, zeroshot_min_similarity
            )
            st.session_state.topic_models = topic_models
            st.session_state.doc_groups = doc_groups
            st.session_state.emb_groups = emb_groups
            st.success("Model training complete!")
            save_models()
        
        # Display Results
        if 'topic_models' in st.session_state:
            topic_models = st.session_state.topic_models
            
            plot_num_topics_and_outliers(topic_models)
            plot_topics_per_timestamp(topic_models)
            
            # Display weak signal trend
            if zeroshot_topic_list is not None:
                weak_signal_trends = detect_weak_signals_zeroshot(topic_models, zeroshot_topic_list)
                with st.expander("Zero-shot Weak Signal Trends", expanded=True):
                    fig_trend = go.Figure()
                    for topic, weak_signal_trend in weak_signal_trends.items():
                        timestamps = list(weak_signal_trend.keys())
                        popularity = [weak_signal_trend[timestamp]['Document_Count'] for timestamp in timestamps]
                        hovertext = [f"Topic: {topic}<br>Timestamp: {timestamp}<br>Popularity: {weak_signal_trend[timestamp]['Document_Count']}" for timestamp in timestamps]
                        fig_trend.add_trace(go.Scatter(x=timestamps, y=popularity, mode='lines+markers', name=topic, hovertext=hovertext, hoverinfo='text'))
                    fig_trend.update_layout(title="Popularity of Zero-Shot Topics", xaxis_title="Timestamp", yaxis_title="Popularity")
                    st.plotly_chart(fig_trend, use_container_width=True)
        
        # Merge models button
        if st.button("Merge Models"):
            with st.spinner("Merging models..."):
                topic_dfs = {}
                for period, topic_model in st.session_state.topic_models.items():
                    docs = st.session_state.doc_groups[period]
                    embeddings = st.session_state.emb_groups[period]
                    topic_dfs[period] = preprocess_model(topic_model, docs, embeddings)

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
                        if not (df1.empty or df2.empty):
                            merged_df_without_outliers, merge_history, new_topics = merge_models(df1, 
                                                                                                df2, 
                                                                                                min_similarity=min_similarity, 
                                                                                                timestamp=current_timestamp)
                    elif not df2.empty:
                        merged_df_without_outliers, merge_history, new_topics = merge_models(merged_df_without_outliers, 
                                                                                             df2, 
                                                                                             min_similarity=min_similarity, 
                                                                                             timestamp=current_timestamp)
                    else:
                        continue
                                    
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
        
        if "all_merge_histories_df" in st.session_state:
            plot_topic_size_evolution(st.session_state.all_merge_histories_df, granularity)

            # Create a text input field and a button for taking a closer look at a topic
            topic_number = st.text_input("Enter a topic number to take a closer look:")

            if st.button("Analyze signal"):
                topic_merge_rows = st.session_state.all_merge_histories_df[st.session_state.all_merge_histories_df['Topic1'] == int(topic_number)].sort_values('Timestamp')
                
                if not topic_merge_rows.empty:
                    # Generate a summary using OpenAI ChatGPT
                    content_summary = ""
                    for i, row in enumerate(topic_merge_rows.itertuples()):
                        timestamp = row.Timestamp
                        next_timestamp = timestamp + pd.Timedelta(days=granularity)
                        representation1 = row.Representation1
                        representation2 = row.Representation2
                        documents1 = set(row.Documents1)
                        documents2 = set(row.Documents2)
                        
                        timestamp_str = timestamp.strftime("%Y-%m-%d")
                        next_timestamp_str = next_timestamp.strftime("%Y-%m-%d")
                        
                        content_summary += f"Timestamp: {timestamp_str}\nTopic representation: {representation1}\n"
                        for doc in documents1:
                            content_summary += f"- {doc}\n"
                        content_summary += f"\nTimestamp: {next_timestamp_str}\nTopic representation: {representation2}\n"
                        for doc in documents2:
                            content_summary += f"- {doc}\n"
                        content_summary += "\n"
                    
                    # Generate the summary using OpenAI ChatGPT
                    prompt = get_prompt(language, topic_number, content_summary)
                    
                    with st.spinner("Generating summary..."):
                        try:
                            completion = client.chat.completions.create(
                                model="gpt-3.5-turbo-0125",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant, skilled in summarizing topic evolution over time."},
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            summary = completion.choices[0].message.content
                            st.markdown(summary)
                        except:
                            st.warning("Unable to generate a summary. Too many documents.")
                else:
                    st.warning(f"Topic {topic_number} not found in the merge histories.")    
            # Create the Sankey Diagram
            create_sankey_diagram(st.session_state.all_merge_histories_df)
            
            
        
        if "all_new_topics_df" in st.session_state and not st.session_state.all_new_topics_df.empty:
            plot_newly_emerged_topics(st.session_state.all_new_topics_df)


if __name__ == "__main__":

    client = OpenAI(api_key="sk-proj-ouL9Jgzzb59Z0wCSt5khT3BlbkFJDb9WDt0Lu4aZqi8M1YxH")    
    main()