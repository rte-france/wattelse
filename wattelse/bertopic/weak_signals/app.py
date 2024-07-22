import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import shutil
from loguru import logger
from openai import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm 

from session_state_manager import SessionStateManager
from data_loading import load_and_preprocess_data, group_by_days
from topic_modeling import train_topic_models, merge_models, preprocess_model
from visualizations import (plot_num_topics_and_outliers, plot_topics_per_timestamp, 
                            plot_topic_size_evolution, create_topic_size_evolution_figure, 
                            plot_newly_emerged_topics, create_sankey_diagram)
from weak_signals import detect_weak_signals_zeroshot, calculate_signal_popularity
from prompts import get_prompt
from global_vars import *
from nltk.corpus import stopwords
import glob

def save_state():
    os.makedirs(CACHE_DIR, exist_ok=True)
    state_file = CACHE_DIR / STATE_FILE
    embeddings_file = CACHE_DIR / EMBEDDINGS_FILE

    state = SessionStateManager.get_multiple(
        'selected_file', 'min_chars', 'split_by_paragraph', 'timeframe_slider',
        'language', 'embedding_model_name', 'embedding_model', 'sample_size',
        'min_similarity', 'zeroshot_min_similarity', 'embedding_dtype'
    )

    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success("Application state saved.")

def restore_state():
    state_file = CACHE_DIR / STATE_FILE
    embeddings_file = CACHE_DIR / EMBEDDINGS_FILE

    if state_file.exists() and embeddings_file.exists():
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        SessionStateManager.set_multiple(**state)
        SessionStateManager.set('embeddings', np.load(embeddings_file))
        st.success("Application state restored.")
    else:
        st.warning("No saved state found.")

def save_models():
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    topic_models = SessionStateManager.get('topic_models', {})
    for period, topic_model in topic_models.items():
        model_dir = MODELS_DIR / period.strftime("%Y-%m-%d")
        model_dir.mkdir(exist_ok=True)
        embedding_model = SessionStateManager.get('embedding_model')
        topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

        topic_model.doc_info_df.to_pickle(model_dir / DOC_INFO_DF_FILE)
        topic_model.topic_info_df.to_pickle(model_dir / TOPIC_INFO_DF_FILE)

    with open(CACHE_DIR / DOC_GROUPS_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('doc_groups'), f)
    with open(CACHE_DIR / EMB_GROUPS_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('emb_groups'), f)
    with open(CACHE_DIR / GRANULARITY_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('granularity_select'), f)

    hyperparams = SessionStateManager.get_multiple(
        'umap_n_components', 'umap_n_neighbors', 'hdbscan_min_cluster_size',
        'hdbscan_min_samples', 'hdbscan_cluster_selection_method', 'top_n_words',
        'vectorizer_ngram_range', 'min_df'
    )
    with open(CACHE_DIR / HYPERPARAMS_FILE, 'wb') as f:
        pickle.dump(hyperparams, f)

    st.success("Models saved.")

def restore_models():
    if not MODELS_DIR.exists():
        st.warning("No saved models found.")
        return

    topic_models = {}
    for period_dir in MODELS_DIR.iterdir():
        if period_dir.is_dir():
            topic_model = BERTopic.load(period_dir)
            
            doc_info_df_file = period_dir / DOC_INFO_DF_FILE
            topic_info_df_file = period_dir / TOPIC_INFO_DF_FILE
            if doc_info_df_file.exists() and topic_info_df_file.exists():
                topic_model.doc_info_df = pd.read_pickle(doc_info_df_file)
                topic_model.topic_info_df = pd.read_pickle(topic_info_df_file)
            else:
                logger.warning(f"doc_info_df or topic_info_df not found for period {period_dir.name}")

            period = pd.Timestamp(period_dir.name.replace('_', ':'))
            topic_models[period] = topic_model

    SessionStateManager.set('topic_models', topic_models)

    for file, key in [(DOC_GROUPS_FILE, 'doc_groups'), (EMB_GROUPS_FILE, 'emb_groups')]:
        file_path = CACHE_DIR / file
        if file_path.exists():
            with open(file_path, 'rb') as f:
                SessionStateManager.set(key, pickle.load(f))
        else:
            logger.warning(f"{file} not found.")

    granularity_file = CACHE_DIR / GRANULARITY_FILE
    if granularity_file.exists():
        with open(granularity_file, 'rb') as f:
            SessionStateManager.set('granularity_select', pickle.load(f))
    else:
        logger.warning("Granularity value not found.")

    hyperparams_file = CACHE_DIR / HYPERPARAMS_FILE
    if hyperparams_file.exists():
        with open(hyperparams_file, 'rb') as f:
            SessionStateManager.set_multiple(**pickle.load(f))
    else:
        logger.warning("Hyperparameters file not found.")

    st.success("Models restored.")


def purge_cache():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success(f"Cache purged.")
    else:
        st.warning(f"No cache found.")

def main():
    # Restore Previous Run and Purge Cache buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restore Previous Run"):
            restore_state()
            restore_models()
    with col2:
        if st.button("Purge Cache"):
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                st.success("Cache purged.")
            else:
                st.warning("No cache found.")

    # Sidebar menu for BERTopic hyperparameters
    st.sidebar.header("BERTopic Hyperparameters")
    with st.sidebar.expander("Embedding Model Settings", expanded=True):
        language = st.sidebar.selectbox("Select Language", LANGUAGES, key='language')
        embedding_dtype = st.selectbox("Embedding Dtype", EMBEDDING_DTYPES, key='embedding_dtype')
        
        embedding_models = ENGLISH_EMBEDDING_MODELS if language == "English" else FRENCH_EMBEDDING_MODELS
        embedding_model_name = st.sidebar.selectbox("Embedding Model", embedding_models, key='embedding_model_name')

    # UMAP, HDBSCAN, and Vectorizer Hyperparameters
    for expander, params in [
        ("UMAP Hyperparameters", [
            ("umap_n_components", "UMAP n_components", DEFAULT_UMAP_N_COMPONENTS, 2, 100),
            ("umap_n_neighbors", "UMAP n_neighbors", DEFAULT_UMAP_N_NEIGHBORS, 2, 100)
        ]),
        ("HDBSCAN Hyperparameters", [
            ("hdbscan_min_cluster_size", "HDBSCAN min_cluster_size", DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE, 2, 100),
            ("hdbscan_min_samples", "HDBSCAN min_sample", DEFAULT_HDBSCAN_MIN_SAMPLES, 1, 100)
        ]),
        ("Vectorizer Hyperparameters", [
            ("top_n_words", "Top N Words", DEFAULT_TOP_N_WORDS, 1, 50),
            ("min_df", "min_df", DEFAULT_MIN_DF, 1, 50)
        ])
    ]:
        with st.sidebar.expander(expander, expanded=True):
            for key, label, default, min_val, max_val in params:
                st.number_input(label, value=default, min_value=min_val, max_value=max_val, key=key)
            
            if expander == "HDBSCAN Hyperparameters":
                st.selectbox("Cluster Selection Method", HDBSCAN_CLUSTER_SELECTION_METHODS, key='hdbscan_cluster_selection_method')
            elif expander == "Vectorizer Hyperparameters":
                st.selectbox("N-Gram range", VECTORIZER_NGRAM_RANGES, key='vectorizer_ngram_range')

    with st.sidebar.expander("Merging Hyperparameters", expanded=True):
        st.slider("Minimum Similarity for Merging", 0.0, 1.0, DEFAULT_MIN_SIMILARITY, 0.01, key='min_similarity')

    with st.sidebar.expander("Zero-shot Parameters", expanded=True):
        st.slider("Zeroshot Minimum Similarity", 0.0, 1.0, DEFAULT_ZEROSHOT_MIN_SIMILARITY, 0.01, key='zeroshot_min_similarity')

    # Load and preprocess data
    selected_file = st.selectbox("Select a dataset", [(f.name, f.suffix[1:]) for f in DATA_PATH.glob('*')], key='selected_file')
    min_chars = st.number_input("Minimum Characters", value=MIN_CHARS_DEFAULT, min_value=0, max_value=1000, key='min_chars')
    split_by_paragraph = st.checkbox("Split text by paragraphs", value=False, key="split_by_paragraph")

    df = load_and_preprocess_data(selected_file, language, min_chars, split_by_paragraph)

    # Select timeframe
    min_date, max_date = df['timestamp'].dt.date.agg(['min', 'max'])
    start_date, end_date = st.slider("Select Timeframe", min_value=min_date, max_value=max_date, value=(min_date, max_date), key='timeframe_slider')

    # Filter and sample the DataFrame
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].drop_duplicates(subset='text', keep='first')
    df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

    sample_size = st.number_input("Sample Size", value=SAMPLE_SIZE_DEFAULT or len(df_filtered), min_value=1, max_value=len(df_filtered), key='sample_size')
    if sample_size < len(df_filtered):
        df_filtered = df_filtered.sample(n=sample_size, random_state=42)

    df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

    SessionStateManager.set('timefiltered_df', df_filtered)
    st.write(f"Number of documents in selected timeframe: {len(SessionStateManager.get_dataframe('timefiltered_df'))}")

    # Select granularity
    granularity = st.number_input("Select Granularity", value=DEFAULT_GRANULARITY, min_value=1, max_value=30, key='granularity_select')

    # Show documents per grouped timestamp
    with st.expander("Documents per Timestamp", expanded=True):
        grouped_data = group_by_days(SessionStateManager.get_dataframe('timefiltered_df'), day_granularity=granularity)
        non_empty_timestamps = [timestamp for timestamp, group in grouped_data.items() if not group.empty]
        if non_empty_timestamps:
            selected_timestamp = st.select_slider("Select Timestamp", options=non_empty_timestamps, key='timestamp_slider')
            selected_docs = grouped_data[selected_timestamp]
            st.dataframe(selected_docs[['timestamp', 'text', 'document_id', 'source', 'url']], use_container_width=True)
        else:
            st.warning("No data available for the selected granularity.")

    # Zero-shot topic definition
    zeroshot_topic_list = st.text_input("Enter zero-shot topics (separated by /)", value="")
    zeroshot_topic_list = [topic.strip() for topic in zeroshot_topic_list.split("/") if topic.strip()]
    
    # Embed documents
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            embedding_dtype = SessionStateManager.get('embedding_dtype')
            embedding_model_name = SessionStateManager.get('embedding_model_name')
            
            model_kwargs = {}
            if embedding_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif embedding_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16

            embedding_model = SentenceTransformer(embedding_model_name, device=EMBEDDING_DEVICE, trust_remote_code=True, model_kwargs=model_kwargs)
            embedding_model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
            SessionStateManager.set('embedding_model', embedding_model)

            texts = SessionStateManager.get_dataframe('timefiltered_df')['text'].tolist()
            batch_size = EMBEDDING_BATCH_SIZE
            num_batches = (len(texts) + batch_size - 1) // batch_size

            embeddings = []
            for i in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                embeddings.append(batch_embeddings)

            embeddings = np.concatenate(embeddings, axis=0)
            SessionStateManager.set('embeddings', embeddings)

        st.success("Embeddings calculated successfully!")
        save_state()

    # Train models button
    if SessionStateManager.get_dataframe('timefiltered_df') is not None:
        if st.button("Train Models"):
            grouped_data = group_by_days(SessionStateManager.get_dataframe('timefiltered_df'), day_granularity=granularity)
            
            logger.debug(SessionStateManager.get('language'))
            topic_models, doc_groups, emb_groups = train_topic_models(
                grouped_data, 
                SessionStateManager.get('embedding_model'), 
                SessionStateManager.get_embeddings(),
                umap_n_components=SessionStateManager.get('umap_n_components'),
                umap_n_neighbors=SessionStateManager.get('umap_n_neighbors'),
                hdbscan_min_cluster_size=SessionStateManager.get('hdbscan_min_cluster_size'),
                hdbscan_min_samples=SessionStateManager.get('hdbscan_min_samples'),
                hdbscan_cluster_selection_method=SessionStateManager.get('hdbscan_cluster_selection_method'),
                vectorizer_ngram_range=SessionStateManager.get('vectorizer_ngram_range'),
                min_df=SessionStateManager.get('min_df'),
                top_n_words=SessionStateManager.get('top_n_words'),
                zeroshot_topic_list=zeroshot_topic_list,
                zeroshot_min_similarity=SessionStateManager.get('zeroshot_min_similarity'),
                language=SessionStateManager.get('language')
            )
            SessionStateManager.set_multiple(topic_models=topic_models, doc_groups=doc_groups, emb_groups=emb_groups)
            st.success("Model training complete!")
            save_models()
        
        # Display Results
        topic_models = SessionStateManager.get('topic_models')
        if topic_models:
            plot_num_topics_and_outliers(topic_models)
            plot_topics_per_timestamp(topic_models)

            # Display weak signal trend
            if zeroshot_topic_list:
                weak_signal_trends = detect_weak_signals_zeroshot(topic_models, zeroshot_topic_list, granularity)
                with st.expander("Zero-shot Weak Signal Trends", expanded=True):
                    fig_trend = go.Figure()
                    for topic, weak_signal_trend in weak_signal_trends.items():
                        timestamps = list(weak_signal_trend.keys())
                        popularity = [weak_signal_trend[timestamp]['Document_Count'] for timestamp in timestamps]
                        hovertext = [
                            f"Topic: {topic}<br>Timestamp: {timestamp}<br>Popularity: {weak_signal_trend[timestamp]['Document_Count']}<br>Representation: {weak_signal_trend[timestamp]['Representation']}"
                            for timestamp in timestamps
                        ]
                        fig_trend.add_trace(go.Scatter(x=timestamps, y=popularity, mode='lines+markers', name=topic, hovertext=hovertext, hoverinfo='text'))
                    fig_trend.update_layout(title="Popularity of Zero-Shot Topics", xaxis_title="Timestamp", yaxis_title="Popularity")
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Display the dataframe with zeroshot topics information
                    zeroshot_topics_data = [
                        {
                            'Topic': topic,
                            'Timestamp': timestamp,
                            'Representation': data['Representation'],
                            'Representative_Docs': data['Representative_Docs'],
                            'Count': data['Count'],
                            'Document_Count': data['Document_Count']
                        }
                        for topic, weak_signal_trend in weak_signal_trends.items()
                        for timestamp, data in weak_signal_trend.items()
                    ]
                    zeroshot_topics_df = pd.DataFrame(zeroshot_topics_data)
                    st.dataframe(zeroshot_topics_df, use_container_width=True)

                    # Save the zeroshot topics data to a JSON file
                    json_file_path = ZEROSHOT_TOPICS_DATA_DIR / ZEROSHOT_TOPICS_DATA_FILE
                    json_file_path.mkdir(parents=True, exist_ok=True)
                    
                    zeroshot_topics_df.to_json(json_file_path, orient='records', date_format='iso', indent=4)
                    st.success(f"Zeroshot topics data saved to {json_file_path}")
        
        # Merge models button
        if st.button("Merge Models"):
            with st.spinner("Merging models..."):
                topic_dfs = {period: preprocess_model(model, SessionStateManager.get('doc_groups')[period], SessionStateManager.get('emb_groups')[period])
                             for period, model in SessionStateManager.get('topic_models').items()}

                timestamps = sorted(topic_dfs.keys())
                merged_df_without_outliers = None
                all_merge_histories = []
                all_new_topics = []
                
                progress_bar = st.progress(0)
                merge_df_size_over_time = []

                for i, (current_timestamp, next_timestamp) in enumerate(zip(timestamps[:-1], timestamps[1:])):
                    df1 = topic_dfs[current_timestamp][topic_dfs[current_timestamp]['Topic'] != -1]
                    df2 = topic_dfs[next_timestamp][topic_dfs[next_timestamp]['Topic'] != -1]

                    if merged_df_without_outliers is None:
                        if not (df1.empty or df2.empty):
                            merged_df_without_outliers, merge_history, new_topics = merge_models(df1, df2, 
                                                                                                min_similarity=SessionStateManager.get('min_similarity'), 
                                                                                                timestamp=current_timestamp)
                    elif not df2.empty:
                        merged_df_without_outliers, merge_history, new_topics = merge_models(merged_df_without_outliers, df2, 
                                                                                            min_similarity=SessionStateManager.get('min_similarity'), 
                                                                                            timestamp=current_timestamp)
                    else:
                        continue
                                    
                    all_merge_histories.append(merge_history)
                    all_new_topics.append(new_topics)
                    merge_df_size_over_time.append((current_timestamp, merged_df_without_outliers['Topic'].max() + 1))
                    
                    progress_bar.progress((i + 1) / len(timestamps))
                
                all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
                all_new_topics_df = pd.concat(all_new_topics, ignore_index=True)
                
                SessionStateManager.set_multiple(
                    merged_df=merged_df_without_outliers,
                    all_merge_histories_df=all_merge_histories_df,
                    all_new_topics_df=all_new_topics_df
                )

                topic_sizes, topic_last_popularity, topic_last_update = calculate_signal_popularity(all_merge_histories_df, granularity)
                SessionStateManager.set_multiple(
                    topic_sizes=topic_sizes,
                    topic_last_popularity=topic_last_popularity,
                    topic_last_update=topic_last_update
                )

            st.success("Model merging complete!")
            
        # Plot topic size evolution
        if SessionStateManager.get('all_merge_histories_df') is not None:
            window_size = st.number_input("Retrospective Period (days)", min_value=1, max_value=MAX_WINDOW_SIZE, value=DEFAULT_WINDOW_SIZE, key='window_size')

            all_merge_histories_df = SessionStateManager.get('all_merge_histories_df')
            min_datetime = all_merge_histories_df['Timestamp'].min().to_pydatetime()
            max_datetime = all_merge_histories_df['Timestamp'].max().to_pydatetime()

            current_date = st.slider(
                "Current date",
                min_value=min_datetime,
                max_value=max_datetime,
                format="YYYY-MM-DD",
            )

            plot_topic_size_evolution(create_topic_size_evolution_figure(), 
                                      window_size, granularity, current_date, min_datetime, max_datetime)

            # Analyze signal
            topic_number = st.text_input("Enter a topic number to take a closer look:")

            if st.button("Analyze signal"):
                all_merge_histories_df = SessionStateManager.get('all_merge_histories_df')
                topic_merge_rows = all_merge_histories_df[all_merge_histories_df['Topic1'] == int(topic_number)].sort_values('Timestamp')
                topic_merge_rows_filtered = topic_merge_rows[topic_merge_rows['Timestamp'] <= current_date]

                if not topic_merge_rows_filtered.empty:
                    content_summary = "\n".join([
                        f"Timestamp: {row.Timestamp.strftime('%Y-%m-%d')}\n"
                        f"Topic representation: {row.Representation1}\n"
                        f"{' '.join(f'- {doc}' for doc in row.Documents1 if isinstance(doc, str))}\n"
                        f"Timestamp: {(row.Timestamp + pd.Timedelta(days=granularity)).strftime('%Y-%m-%d')}\n"
                        f"Topic representation: {row.Representation2}\n"
                        f"{' '.join(f'- {doc}' for doc in row.Documents2 if isinstance(doc, str))}\n"
                        for row in topic_merge_rows_filtered.itertuples()
                    ])

                    prompt = get_prompt(SessionStateManager.get('language'), topic_number, content_summary)
                    with st.spinner("Generating summary..."):
                        try:
                            completion = client.chat.completions.create(
                                model=GPT_MODEL,
                                messages=[
                                    {"role": "system", "content": GPT_SYSTEM_MESSAGE},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=GPT_TEMPERATURE,
                            )
                            summary = completion.choices[0].message.content
                            st.markdown(summary)
                        except Exception as e:
                            st.error(e)
                else:
                    st.warning(f"Topic {topic_number} not found in the merge histories within the specified window.")

            # Create the Sankey Diagram
            create_sankey_diagram(SessionStateManager.get('all_merge_histories_df'))
            
        if SessionStateManager.get('all_new_topics_df') is not None:
            plot_newly_emerged_topics(SessionStateManager.get('all_new_topics_df'))

        if st.button("Retrieve Topic Counts"):
            topic_counts = [(timestamp, model.topic_info_df['Topic'].max() + 1) 
                            for timestamp, model in SessionStateManager.get('topic_models').items()]
            df = pd.DataFrame(topic_counts, columns=['timestamp', 'num_topics'])
            df2 = pd.DataFrame(SessionStateManager.get('merge_df_size_over_time', []), columns=['timestamp', 'num_topics'])

            json_data = df.to_json(orient='records', date_format='iso', indent=4)
            json_data_2 = df2.to_json(orient='records', date_format='iso', indent=4)

            json_file_path = SIGNAL_EVOLUTION_DATA_DIR / SIGNAL_EVOLUTION_DATA_FILE
            json_file_path.mkdir(parents=True, exist_ok=True)
            (json_file_path).write_text(json_data)

            json_file_path = SIGNAL_EVOLUTION_DATA_DIR / SIGNAL_EVOLUTION_DATA_FILE_2
            json_file_path.mkdir(parents=True, exist_ok=True)
            (json_file_path).write_text(json_data_2)

            st.success(f"Topic and signal counts saved to {json_file_path}")

if __name__ == "__main__":
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT) 
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    main()











































