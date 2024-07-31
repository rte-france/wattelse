import os
import pickle
import shutil

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from loguru import logger

from data_loading import load_and_preprocess_data, group_by_days
from global_vars import *
from session_state_manager import SessionStateManager
from topic_modeling import embed_documents, train_topic_models, merge_models, preprocess_model
from visualizations import (plot_num_topics_and_outliers, plot_topics_per_timestamp,
                            plot_topic_size_evolution, create_topic_size_evolution_figure,
                            plot_newly_emerged_topics, create_sankey_diagram)
from wattelse.bertopic.utils import PLOTLY_BUTTON_SAVE_CONFIG, TEXT_COLUMN
from weak_signals import detect_weak_signals_zeroshot, calculate_signal_popularity, analyze_signal, \
    save_signal_evolution_data


def save_state():
    os.makedirs(CACHE_PATH, exist_ok=True)
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    state = SessionStateManager.get_multiple(
        'selected_file', 'min_chars', 'split_by_paragraph', 'timeframe_slider',
        'language', 'embedding_model_name', 'embedding_model', 'sample_size',
        'min_similarity', 'zeroshot_min_similarity', 'embedding_dtype', 
        'data_embedded'
    )

    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success("Application state saved.")

def restore_state():
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

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
        topic_model.save(model_dir, serialization="safetensors", save_ctfidf=False, save_embedding_model=embedding_model)

        topic_model.doc_info_df.to_pickle(model_dir / DOC_INFO_DF_FILE)
        topic_model.topic_info_df.to_pickle(model_dir / TOPIC_INFO_DF_FILE)

    with open(CACHE_PATH / DOC_GROUPS_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('doc_groups'), f)
    with open(CACHE_PATH / EMB_GROUPS_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('emb_groups'), f)
    with open(CACHE_PATH / GRANULARITY_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('granularity_select'), f)

    # Save the models_trained flag
    with open(CACHE_PATH / MODELS_TRAINED_FILE, 'wb') as f:
        pickle.dump(SessionStateManager.get('models_trained'), f)

    hyperparams = SessionStateManager.get_multiple(
        'umap_n_components', 'umap_n_neighbors', 'hdbscan_min_cluster_size',
        'hdbscan_min_samples', 'hdbscan_cluster_selection_method', 'top_n_words',
        'vectorizer_ngram_range', 'min_df'
    )
    with open(CACHE_PATH / HYPERPARAMS_FILE, 'wb') as f:
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
        file_path = CACHE_PATH / file
        if file_path.exists():
            with open(file_path, 'rb') as f:
                SessionStateManager.set(key, pickle.load(f))
        else:
            logger.warning(f"{file} not found.")

    granularity_file = CACHE_PATH / GRANULARITY_FILE
    if granularity_file.exists():
        with open(granularity_file, 'rb') as f:
            SessionStateManager.set('granularity_select', pickle.load(f))
    else:
        logger.warning("Granularity value not found.")

    # Restore the models_trained flag
    models_trained_file = CACHE_PATH / MODELS_TRAINED_FILE
    if models_trained_file.exists():
        with open(models_trained_file, 'rb') as f:
            SessionStateManager.set('models_trained', pickle.load(f))
    else:
        logger.warning("Models trained flag not found.")

    hyperparams_file = CACHE_PATH / HYPERPARAMS_FILE
    if hyperparams_file.exists():
        with open(hyperparams_file, 'rb') as f:
            SessionStateManager.set_multiple(**pickle.load(f))
    else:
        logger.warning("Hyperparameters file not found.")

    st.success("Models restored.")


def purge_cache():
    cache_dir = CACHE_PATH
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success(f"Cache purged.")
    else:
        st.warning(f"No cache found.")


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT, initial_sidebar_state="expanded")
    
    st.title("Topic Modeling and Weak Signal Detection")
    
    # Set the main flags
    SessionStateManager.get_or_set('data_embedded', False)
    SessionStateManager.get_or_set('models_merged', False)
    SessionStateManager.get_or_set('models_trained', False)

    # Sidebar
    with st.sidebar:
        st.header("Settings and Controls")
        
        # State Management
        st.subheader("State Management")
        
        if st.button("Restore Previous Run", use_container_width=True):
            restore_state()
            restore_models()

        if st.button("Purge Cache", use_container_width=True):
            purge_cache()

        if st.button("Clear session state", use_container_width=True):
            SessionStateManager.clear()
        
        # BERTopic Hyperparameters
        st.subheader("BERTopic Hyperparameters")
        with st.expander("Embedding Model Settings", expanded=False):
            language = st.selectbox("Select Language", LANGUAGES, key='language')
            embedding_dtype = st.selectbox("Embedding Dtype", EMBEDDING_DTYPES, key='embedding_dtype')
            
            embedding_models = ENGLISH_EMBEDDING_MODELS if language == "English" else FRENCH_EMBEDDING_MODELS
            embedding_model_name = st.selectbox("Embedding Model", embedding_models, key='embedding_model_name')

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
            with st.expander(expander, expanded=False):
                for key, label, default, min_val, max_val in params:
                    st.number_input(label, value=default, min_value=min_val, max_value=max_val, key=key)
                
                if expander == "HDBSCAN Hyperparameters":
                    st.selectbox("Cluster Selection Method", HDBSCAN_CLUSTER_SELECTION_METHODS, key='hdbscan_cluster_selection_method')
                elif expander == "Vectorizer Hyperparameters":
                    st.selectbox("N-Gram range", VECTORIZER_NGRAM_RANGES, key='vectorizer_ngram_range')

        with st.expander("Merging Hyperparameters", expanded=False):
            st.slider("Minimum Similarity for Merging", 0.0, 1.0, DEFAULT_MIN_SIMILARITY, 0.01, key='min_similarity')

        with st.expander("Zero-shot Parameters", expanded=False):
            st.slider("Zeroshot Minimum Similarity", 0.0, 1.0, DEFAULT_ZEROSHOT_MIN_SIMILARITY, 0.01, key='zeroshot_min_similarity')

    # Main content
    tab1, tab2, tab3 = st.tabs(["Data Loading", "Model Training", "Results Analysis"])
    
    with tab1:
        st.header("Data Loading and Preprocessing")
        
        selected_file = st.selectbox("Select a dataset", [(f.name, f.suffix[1:]) for f in DATA_PATH.glob('*')], key='selected_file')
        col1, col2 = st.columns(2)
        with col1:
            min_chars = st.number_input("Minimum Characters", value=MIN_CHARS_DEFAULT, min_value=0, max_value=1000, key='min_chars')
        with col2:
            split_by_paragraph = st.checkbox("Split text by paragraphs", value=False, key="split_by_paragraph")

        df = load_and_preprocess_data(selected_file, language, min_chars, split_by_paragraph)

        # Select timeframe
        min_date, max_date = df['timestamp'].dt.date.agg(['min', 'max'])
        start_date, end_date = st.slider("Select Timeframe", min_value=min_date, max_value=max_date, value=(min_date, max_date), key='timeframe_slider')

        # Filter and sample the DataFrame
        df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)].drop_duplicates(subset=TEXT_COLUMN, keep='first')
        df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

        sample_size = st.number_input("Sample Size", value=SAMPLE_SIZE_DEFAULT or len(df_filtered), min_value=1, max_value=len(df_filtered), key='sample_size')
        if sample_size < len(df_filtered):
            df_filtered = df_filtered.sample(n=sample_size, random_state=42)

        df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

        SessionStateManager.set('timefiltered_df', df_filtered)
        st.write(f"Number of documents in selected timeframe: {len(SessionStateManager.get_dataframe('timefiltered_df'))}")
        st.dataframe(SessionStateManager.get_dataframe('timefiltered_df')[[TEXT_COLUMN, 'timestamp']], use_container_width=True)

        # Embed documents
        if st.button("Embed Documents"):
            with st.spinner("Embedding documents..."):
                embedding_dtype = SessionStateManager.get('embedding_dtype')
                embedding_model_name = SessionStateManager.get('embedding_model_name')
                
                texts = SessionStateManager.get_dataframe('timefiltered_df')[TEXT_COLUMN].tolist()
                
                try:
                    embedding_model, embeddings = embed_documents(
                        texts=texts,
                        embedding_model_name=embedding_model_name,
                        embedding_dtype=embedding_dtype,
                        embedding_device=EMBEDDING_DEVICE,
                        batch_size=EMBEDDING_BATCH_SIZE,
                        max_seq_length=EMBEDDING_MAX_SEQ_LENGTH
                    )
                    
                    SessionStateManager.set('embedding_model', embedding_model)
                    SessionStateManager.set('embeddings', embeddings)
                    SessionStateManager.set('data_embedded', True)

                    st.success("Embeddings calculated successfully!")
                    save_state()
                except Exception as e:
                    st.error(f"An error occurred while embedding documents: {str(e)}")
           
    with tab2:
        st.header("Model Training")

        # Select granularity
        granularity = st.number_input("Select Granularity", 
                                      value=DEFAULT_GRANULARITY, 
                                      min_value=1, 
                                      max_value=30, 
                                      key='granularity_select',
                                      help='Number of days to split the data by')

        # Show documents per grouped timestamp
        with st.expander("Documents per Timestamp", expanded=True):
            grouped_data = group_by_days(SessionStateManager.get_dataframe('timefiltered_df'), day_granularity=granularity)
            non_empty_timestamps = [timestamp for timestamp, group in grouped_data.items() if not group.empty]
            if non_empty_timestamps:
                selected_timestamp = st.select_slider("Select Timestamp", options=non_empty_timestamps, key='timestamp_slider')
                selected_docs = grouped_data[selected_timestamp]
                st.dataframe(selected_docs[['timestamp', TEXT_COLUMN, 'document_id', 'source', 'url']], use_container_width=True)
            else:
                st.warning("No data available for the selected granularity.")
        
        if not SessionStateManager.get('data_embedded', False):
            st.warning("Please embed data before proceeding to model training.")
            st.stop()
        else:
            # Zero-shot topic definition
            zeroshot_topic_list = st.text_input("Enter zero-shot topics (separated by /)", value="")
            zeroshot_topic_list = [topic.strip() for topic in zeroshot_topic_list.split("/") if topic.strip()]

            if st.button("Train Models"):
                with st.spinner("Training models..."):
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
                    SessionStateManager.set('models_trained', True)
                    st.success("Model training complete!")
                    save_models()

            if not SessionStateManager.get('models_trained', False):
                st.stop()
            else:

                if st.button("Merge Models"):
                    with st.spinner("Merging models..."):
                        topic_dfs = {period: preprocess_model(model, SessionStateManager.get('doc_groups')[period], SessionStateManager.get('emb_groups')[period])
                                    for period, model in SessionStateManager.get('topic_models').items()}

                        timestamps = sorted(topic_dfs.keys())
                        merged_df_without_outliers = None
                        all_merge_histories = []
                        all_new_topics = []
                        
                        progress_bar = st.progress(0)
                        SessionStateManager.set('merge_df_size_over_time', [])


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
                            merge_df_size_over_time = SessionStateManager.get('merge_df_size_over_time')
                            merge_df_size_over_time.append((current_timestamp, merged_df_without_outliers['Topic'].max() + 1))
                            SessionStateManager.update('merge_df_size_over_time', merge_df_size_over_time)

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

                        SessionStateManager.set('models_merged', True)

                    st.success("Model merging complete!")

    with tab3:
        st.header("Results Analysis")


        if not SessionStateManager.get('data_embedded', False):
            st.warning("Please embed data and train models before proceeding to analysis.")
            st.stop() 

        elif not SessionStateManager.get('models_trained', False):
            st.warning("Please train models before proceeding to analysis.")
            st.stop() 

        else:
            topic_models = SessionStateManager.get('topic_models')
            st.subheader("Topic Overview")
            plot_num_topics_and_outliers(topic_models)
            plot_topics_per_timestamp(topic_models)

            # Display zeroshot signal trend
            if zeroshot_topic_list:
                st.subheader("Zero-shot Weak Signal Trends")
                weak_signal_trends = detect_weak_signals_zeroshot(topic_models, zeroshot_topic_list, granularity)
                with st.expander("Zero-shot Weak Signal Trends", expanded=False):
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
                    st.plotly_chart(fig_trend, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)

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
                    json_file_path = ZEROSHOT_TOPICS_DATA_DIR
                    json_file_path.mkdir(parents=True, exist_ok=True)
                    
                    zeroshot_topics_df.to_json(json_file_path / ZEROSHOT_TOPICS_DATA_FILE, orient='records', date_format='iso', indent=4)
                    st.success(f"Zeroshot topics data saved to {json_file_path}")

            if not SessionStateManager.get('models_merged', False):
                st.warning("Please merge models to view additional analyses.")
                st.stop()

            else:
                # Display merged signal trend
                st.subheader("Topic Size Evolution")
                st.dataframe(SessionStateManager.get('all_merge_histories_df')[['Timestamp', 'Topic1', 'Topic2', 'Representation1', 'Representation2', 'Document_Count1', 'Document_Count2']])

                with st.expander("Topic Popularity Evolution", expanded=True):
                    window_size = st.number_input("Retrospective Period (days)", min_value=1, max_value=MAX_WINDOW_SIZE, value=DEFAULT_WINDOW_SIZE, key='window_size')

                    all_merge_histories_df = SessionStateManager.get('all_merge_histories_df')
                    min_datetime = all_merge_histories_df['Timestamp'].min().to_pydatetime()
                    max_datetime = all_merge_histories_df['Timestamp'].max().to_pydatetime()

                    current_date = st.slider(
                        "Current date",
                        min_value=min_datetime,
                        max_value=max_datetime,
                        step=pd.Timedelta(days=granularity),
                        format="YYYY-MM-DD",

                        help="""The earliest selectable date corresponds to the earliest timestamp when topics were merged 
                        (with the smallest possible value being the earliest timestamp in the provided data). 
                        The latest selectable date corresponds to the most recent topic merges, which is at most equal 
                        to the latest timestamp in the data minus the provided granularity."""
                    )

                    plot_topic_size_evolution(create_topic_size_evolution_figure(), 
                                            window_size, granularity, current_date, min_datetime, max_datetime)
                    
                    # Save Signal Evolution Data to investigate later on in a separate notebook
                    start_date, end_date = st.select_slider(
                        "Select date range for saving signal evolution data:",
                        options=pd.date_range(start=min_datetime, end=max_datetime, freq=pd.Timedelta(days=granularity)),
                        value=(min_datetime, max_datetime),
                        format_func=lambda x: x.strftime('%Y-%m-%d'),
                    )

                    if st.button("Save Signal Evolution Data"):
                        try:
                            save_path = save_signal_evolution_data(
                                all_merge_histories_df=all_merge_histories_df,
                                topic_sizes=dict(SessionStateManager.get('topic_sizes')),
                                topic_last_popularity=SessionStateManager.get('topic_last_popularity'),
                                topic_last_update=SessionStateManager.get('topic_last_update'),
                                window_size=SessionStateManager.get('window_size'),
                                granularity=granularity,
                                start_timestamp=pd.Timestamp(start_date),
                                end_timestamp=pd.Timestamp(end_date)
                            )
                            st.success(f"Signal evolution data saved successfully at {save_path}")
                        except Exception as e:
                            st.error(f"Error encountered while saving signal evolution data: {e}")

                # Analyze signal
                st.subheader("Signal Analysis")
                topic_number = st.number_input("Enter a topic number to take a closer look:",
                                               min_value=0,
                                               step=1)

                if st.button("Analyze signal"):
                    try:
                        language = SessionStateManager.get('language')
                        with st.container(height=500, border=True):
                            with st.spinner("Analyzing signal..."):
                                summary, analysis = analyze_signal(topic_number, current_date, all_merge_histories_df, granularity, language)
                                col1, col2 = st.columns(spec=[.5, .5], gap="medium")
                                with col1:
                                    st.markdown(summary)
                                with col2:
                                    st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error while trying to generate signal summary : {e}")

                # Create the Sankey Diagram
                st.subheader("Topic Evolution")
                create_sankey_diagram(SessionStateManager.get('all_merge_histories_df'))
                
                if SessionStateManager.get('all_new_topics_df') is not None:
                    st.subheader("Newly Emerged Topics")
                    plot_newly_emerged_topics(SessionStateManager.get('all_new_topics_df'))

                if st.button("Retrieve Topic Counts"):
                    with st.spinner("Retrieving topic counts..."):
                        # Number of topics per individual topic model
                        individual_model_topic_counts = [(timestamp, model.topic_info_df['Topic'].max() + 1) 
                                                        for timestamp, model in SessionStateManager.get('topic_models').items()]
                        df_individual_models = pd.DataFrame(individual_model_topic_counts, columns=['timestamp', 'num_topics'])

                        # Number of topics per cumulative merged model
                        cumulative_merged_topic_counts = SessionStateManager.get('merge_df_size_over_time', [])
                        df_cumulative_merged = pd.DataFrame(cumulative_merged_topic_counts, columns=['timestamp', 'num_topics'])

                        # Convert to JSON
                        json_individual_models = df_individual_models.to_json(orient='records', date_format='iso', indent=4)
                        json_cumulative_merged = df_cumulative_merged.to_json(orient='records', date_format='iso', indent=4)

                        # Save individual model topic counts
                        json_file_path = SIGNAL_EVOLUTION_DATA_DIR
                        json_file_path.mkdir(parents=True, exist_ok=True)
                        (json_file_path / INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE).write_text(json_individual_models)

                        # Save cumulative merged model topic counts
                        (json_file_path / CUMULATIVE_MERGED_TOPIC_COUNTS_FILE).write_text(json_cumulative_merged)

                        st.success(f"Topic counts for individual and cumulative merged models saved to {json_file_path}")

if __name__ == "__main__":
    main()











































