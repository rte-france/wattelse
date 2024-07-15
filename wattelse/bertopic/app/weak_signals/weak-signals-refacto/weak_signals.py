from typing import Dict, List, Tuple, Any
import pandas as pd
from bertopic import BERTopic
import numpy as np
import os
import pickle
from collections import defaultdict
import streamlit as st
from loguru import logger
from pprint import pprint
from tqdm import tqdm  # Import the tqdm library
import scipy

# def detect_weak_signals_zeroshot(topic_models: Dict[pd.Timestamp, BERTopic], zeroshot_topic_list: List[str]) -> Dict[str, Dict[pd.Timestamp, Dict[str, any]]]:
#     """
#     Detect weak signals based on the zero-shot list of topics to monitor.

#     Args:
#         topic_models (Dict[pd.Timestamp, BERTopic]): Dictionary of BERTopic models for each timestamp.
#         zeroshot_topic_list (List[str]): List of topics to monitor for weak signals.

#     Returns:
#         Dict[str, Dict[pd.Timestamp, Dict[str, any]]]: Dictionary of weak signal trends for each monitored topic.
#     """
#     weak_signal_trends = {}

#     for topic in zeroshot_topic_list:
#         weak_signal_trends[topic] = {}

#         for timestamp, topic_model in topic_models.items():
#             topic_info = topic_model.topic_info_df

#             for _, row in topic_info.iterrows():
#                 if row['Name'] == topic:
#                     weak_signal_trends[topic][timestamp] = {
#                         'Representation': row['Representation'],
#                         'Representative_Docs': row['Representative_Docs'],
#                         'Count': row['Count'],
#                         'Document_Count': row['Document_Count']
#                     }
#                     break

#     return weak_signal_trends

def detect_weak_signals_zeroshot(topic_models: Dict[pd.Timestamp, BERTopic], zeroshot_topic_list: List[str], granularity: int, decay_factor: float = 0.01, decay_power: float = 2) -> Dict[str, Dict[pd.Timestamp, Dict[str, any]]]:
    """
    Detect weak signals based on the zero-shot list of topics to monitor.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): Dictionary of BERTopic models for each timestamp.
        zeroshot_topic_list (List[str]): List of topics to monitor for weak signals.
        granularity (int): The granularity of the timestamps in days.
        decay_factor (float): The decay factor for exponential decay.
        decay_power (float): The decay power for exponential decay.

    Returns:
        Dict[str, Dict[pd.Timestamp, Dict[str, any]]]: Dictionary of weak signal trends for each monitored topic.
    """
    weak_signal_trends = {}

    min_timestamp = min(topic_models.keys())
    max_timestamp = max(topic_models.keys())
    timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq=pd.Timedelta(days=granularity))

    for topic in zeroshot_topic_list:
        weak_signal_trends[topic] = {}
        topic_last_popularity = {}
        topic_last_update = {}

        for timestamp in timestamps:
            if timestamp in topic_models:
                topic_info = topic_models[timestamp].topic_info_df

                topic_data = topic_info[topic_info['Name'] == topic]

                if not topic_data.empty:
                    representation = topic_data['Representation'].values[0]
                    representative_docs = topic_data['Representative_Docs'].values[0]
                    count = topic_data['Count'].values[0]
                    document_count = topic_data['Document_Count'].values[0]

                    if topic not in topic_last_popularity:
                        topic_last_popularity[topic] = document_count
                        topic_last_update[topic] = timestamp

                        weak_signal_trends[topic][timestamp] = {
                            'Representation': representation,
                            'Representative_Docs': representative_docs,
                            'Count': count,
                            'Document_Count': document_count
                        }
                    else:
                        weak_signal_trends[topic][timestamp] = {
                            'Representation': representation,
                            'Representative_Docs': representative_docs,
                            'Count': count,
                            'Document_Count': topic_last_popularity[topic] + document_count
                        }
                        topic_last_popularity[topic] = topic_last_popularity[topic] + document_count
                        topic_last_update[topic] = timestamp
                else:
                    last_popularity = topic_last_popularity.get(topic, 0)
                    last_update = topic_last_update.get(topic, timestamp)

                    time_diff = timestamp - last_update
                    periods_since_last_update = time_diff // pd.Timedelta(days=granularity)

                    decayed_popularity = last_popularity * np.exp(-decay_factor * (periods_since_last_update ** decay_power))

                    weak_signal_trends[topic][timestamp] = {
                        'Representation': None,
                        'Representative_Docs': None,
                        'Count': 0,
                        'Document_Count': decayed_popularity
                    }
                    topic_last_popularity[topic] = decayed_popularity
            else:
                last_popularity = topic_last_popularity.get(topic, 0)
                last_update = topic_last_update.get(topic, timestamp)

                time_diff = timestamp - last_update
                periods_since_last_update = time_diff // pd.Timedelta(days=granularity)

                decayed_popularity = last_popularity * np.exp(-decay_factor * (periods_since_last_update ** decay_power))

                weak_signal_trends[topic][timestamp] = {
                    'Representation': None,
                    'Representative_Docs': None,
                    'Count': 0,
                    'Document_Count': decayed_popularity
                }
                topic_last_popularity[topic] = decayed_popularity

    return weak_signal_trends


########################################################################################################################

def calculate_signal_popularity(all_merge_histories_df: pd.DataFrame, granularity: int, decay_factor: float = 0.01, decay_power: float = 2) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], Dict[int, pd.Timestamp]]:
    """
    Calculate the popularity of signals over time.
    
    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
        granularity (int): The granularity of the timestamps in days.
        decay_factor (float): The decay factor for exponential decay.
        decay_power (float): The decay power for exponential decay.
    
    Returns:
        Tuple[Dict[int, Dict[str, Any]], Dict[int, float], Dict[int, pd.Timestamp]]:
            - topic_sizes: Dictionary storing topic sizes and related information.
            - topic_last_popularity: Dictionary storing the last popularity of each topic.
            - topic_last_update: Dictionary storing the last update timestamp of each topic.
    """
    topic_sizes = defaultdict(lambda: defaultdict(list))
    topic_last_popularity = {}
    topic_last_update = {}

    min_timestamp = all_merge_histories_df['Timestamp'].min()
    max_timestamp = all_merge_histories_df['Timestamp'].max()

    min_datetime = min_timestamp.to_pydatetime()
    max_datetime = max_timestamp.to_pydatetime()

    granularity_timedelta = pd.Timedelta(days=granularity)

    # Create a range of timestamps from min_datetime to max_datetime with the specified granularity
    timestamps = pd.date_range(start=min_datetime, end=max_datetime, freq=granularity_timedelta)

    # Iterate over each timestamp
    for current_timestamp in timestamps:
        # Filter the merge history DataFrame for the current timestamp
        current_df = all_merge_histories_df[all_merge_histories_df['Timestamp'] == current_timestamp]
        
        # Iterate over each topic at the current timestamp
        for _, row in current_df.iterrows():
            current_topic = row['Topic1']

            if current_topic not in topic_sizes:
                # Initialize the topic's data with the first point corresponding to Timestamp1 and popularity Document_Count1
                topic_sizes[current_topic]['Timestamps'].append(current_timestamp)
                topic_sizes[current_topic]['Popularity'].append(row['Document_Count1'])
                topic_sizes[current_topic]['Representation'] = '_'.join(row['Representation1'])
                topic_sizes[current_topic]['Documents'].append((current_timestamp, row['Documents1']))
                topic_sizes[current_topic]['Sources'].append((current_timestamp, row['Source1']))
                topic_sizes[current_topic]['Docs_Count'].append(row['Document_Count1'])
                topic_sizes[current_topic]['Paragraphs_Count'].append(row['Count1'])
                topic_sizes[current_topic]['Source_Diversity'].append(len(set(row['Source1'])))
                topic_sizes[current_topic]['Representations'].append(topic_sizes[current_topic]['Representation'])

                # Update the topic's last popularity and last update timestamp
                topic_last_popularity[current_topic] = row['Document_Count1']
                topic_last_update[current_topic] = current_timestamp
            
            # Update the topic's data with the new timestamp and aggregated popularity
            next_timestamp = current_timestamp + granularity_timedelta
            
            topic_sizes[current_topic]['Timestamps'].append(next_timestamp)
            topic_sizes[current_topic]['Popularity'].append(topic_last_popularity[current_topic] + row['Document_Count2'])
            topic_sizes[current_topic]['Representation'] = '_'.join(row['Representation2'])
            topic_sizes[current_topic]['Documents'].append((next_timestamp, row['Documents2']))
            topic_sizes[current_topic]['Sources'].append((next_timestamp, row['Source2']))
            topic_sizes[current_topic]['Docs_Count'].append(topic_sizes[current_topic]['Docs_Count'][-1] + row['Document_Count2'])
            topic_sizes[current_topic]['Paragraphs_Count'].append(topic_sizes[current_topic]['Paragraphs_Count'][-1] + row['Count2'])
            topic_sizes[current_topic]['Representations'].append(topic_sizes[current_topic]['Representation'])
            all_sources = [source for timestamp, sources in topic_sizes[current_topic]['Sources'] for source in sources]
            all_sources.extend(row['Source2'])
            topic_sizes[current_topic]['Source_Diversity'].append(len(set(all_sources)))

            # Update the topic's last popularity and last update timestamp
            topic_last_popularity[current_topic] = topic_last_popularity[current_topic] + row['Document_Count2']
            topic_last_update[current_topic] = current_timestamp

        # Get the list of topics that have been seen before but not in the current timestamp
        topics_to_decay = set(topic_last_update.keys()) - set(current_df['Topic1'])
        
        # Apply exponential decay to the topics not seen in the current timestamp
        for topic in topics_to_decay:
            last_popularity = topic_last_popularity[topic]
            last_update = topic_last_update[topic]
            
            time_diff = current_timestamp - last_update
            periods_since_last_update = time_diff // granularity_timedelta
            
            decayed_popularity = last_popularity * np.exp(-decay_factor * (periods_since_last_update ** decay_power))
            
            topic_sizes[topic]['Timestamps'].append(current_timestamp)
            topic_sizes[topic]['Popularity'].append(decayed_popularity)
            topic_sizes[topic]['Docs_Count'].append(topic_sizes[topic]['Docs_Count'][-1])
            topic_sizes[topic]['Paragraphs_Count'].append(topic_sizes[topic]['Paragraphs_Count'][-1])
            topic_sizes[topic]['Source_Diversity'].append(topic_sizes[topic]['Source_Diversity'][-1])
            topic_sizes[topic]['Representations'].append(topic_sizes[topic]['Representation'])

            topic_last_popularity[topic] = decayed_popularity

    return topic_sizes, topic_last_popularity, topic_last_update


########################################################################################################################

def classify_signals(topic_sizes: Dict[int, Dict[str, Any]], window_start: pd.Timestamp, window_end: pd.Timestamp, q1: float, q3: float, rising_popularity_only: bool = True, keep_documents: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify signals into weak signal and strong signal dataframes.
    
    Args:
        topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information.
        window_start (pd.Timestamp): The start timestamp of the window.
        window_end (pd.Timestamp): The end timestamp of the window.
        q1 (float): The 10th percentile of popularity values.
        q3 (float): The 50th percentile of popularity values.
        rising_popularity_only (bool): Whether to consider only rising popularity topics as weak signals.
        keep_documents (bool): Whether to keep track of the documents or not.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - noise_topics_df: DataFrame containing noise topics.
            - weak_signal_topics_df: DataFrame containing weak signal topics.
            - strong_signal_topics_df: DataFrame containing strong signal topics.
    """
    noise_topics = []
    weak_signal_topics = []
    strong_signal_topics = []

    # Sort the topics based on their topic ID
    sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[0])

    for topic, data in sorted_topics:
        # Filter the data based on the window_end (current date)
        filtered_data = {
            'Timestamps': [ts for ts in data['Timestamps'] if ts <= window_end],
            'Popularity': [pop for ts, pop in zip(data['Timestamps'], data['Popularity']) if ts <= window_end],
            'Representation': [rep for ts, rep in zip(data['Timestamps'], data['Representations']) if ts <= window_end],
            'Documents': [doc for ts, docs in data['Documents'] if ts <= window_end for doc in docs] if keep_documents else [],
            'Sources': [sources for ts, sources in data['Sources'] if ts <= window_end],
            'Docs_Count': [count for ts, count in zip(data['Timestamps'], data['Docs_Count']) if ts <= window_end],
            'Paragraphs_Count': [count for ts, count in zip(data['Timestamps'], data['Paragraphs_Count']) if ts <= window_end],
            'Source_Diversity': [div for ts, div in zip(data['Timestamps'], data['Source_Diversity']) if ts <= window_end]
        }

        # Check if the filtered data is empty
        if not filtered_data['Timestamps']:
            continue

        window_popularities = [
            (timestamp, popularity) for timestamp, popularity in zip(filtered_data['Timestamps'], filtered_data['Popularity'])
            if window_start <= timestamp <= window_end
        ]
        
        if window_popularities:
            latest_timestamp, latest_popularity = window_popularities[-1]
            docs_count = filtered_data['Docs_Count'][-1] if filtered_data['Docs_Count'] else 0
            paragraphs_count = filtered_data['Paragraphs_Count'][-1] if filtered_data['Paragraphs_Count'] else 0
            source_diversity = filtered_data['Source_Diversity'][-1] if filtered_data['Source_Diversity'] else 0
            
            if latest_popularity < q1:
                noise_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))
            elif q1 <= latest_popularity <= q3:
                if rising_popularity_only:
                    retrospective_start = latest_timestamp - pd.Timedelta(days=14)
                    retrospective_data = [(timestamp, popularity) for timestamp, popularity in zip(filtered_data['Timestamps'], filtered_data['Popularity'])
                                          if retrospective_start <= timestamp <= latest_timestamp]
                    
                    if len(retrospective_data) >= 2:
                        x = range(len(retrospective_data))
                        y = [popularity for _, popularity in retrospective_data]
                        slope, _, _, _, _ = scipy.stats.linregress(x, y)
                        
                        if slope > 0:
                            weak_signal_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))
                        else:
                            noise_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))
                    else:
                        weak_signal_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))
                else:
                    weak_signal_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))
            else:
                strong_signal_topics.append((topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data))

    # Create DataFrames for each category using list comprehensions
    noise_topics_df = pd.DataFrame([{
        'Topic': topic,
        'Representation': filtered_data['Representation'][-1],
        'Latest_Popularity': latest_popularity,
        'Docs_Count': docs_count,
        'Paragraphs_Count': paragraphs_count,
        'Latest_Timestamp': latest_timestamp,
        'Documents': filtered_data['Documents'] if keep_documents else [],
        'Sources': {source for sources in filtered_data['Sources'] for source in sources},
        'Source_Diversity': source_diversity
    } for topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data in noise_topics])

    weak_signal_topics_df = pd.DataFrame([{
        'Topic': topic,
        'Representation': filtered_data['Representation'][-1],
        'Latest_Popularity': latest_popularity,
        'Docs_Count': docs_count,
        'Paragraphs_Count': paragraphs_count,
        'Latest_Timestamp': latest_timestamp,
        'Documents': filtered_data['Documents'] if keep_documents else [],
        'Sources': {source for sources in filtered_data['Sources'] for source in sources},
        'Source_Diversity': source_diversity
    } for topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data in weak_signal_topics])

    strong_signal_topics_df = pd.DataFrame([{
        'Topic': topic,
        'Representation': filtered_data['Representation'][-1],
        'Latest_Popularity': latest_popularity,
        'Docs_Count': docs_count,
        'Paragraphs_Count': paragraphs_count,
        'Latest_Timestamp': latest_timestamp,
        'Documents': filtered_data['Documents'] if keep_documents else [],
        'Sources': {source for sources in filtered_data['Sources'] for source in sources},
        'Source_Diversity': source_diversity
    } for topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data in strong_signal_topics])

    # Filter out rows with Latest_Popularity < 0.01 using boolean indexing
    noise_topics_df = noise_topics_df[noise_topics_df['Latest_Popularity'] >= 0.01]
    weak_signal_topics_df = weak_signal_topics_df[weak_signal_topics_df['Latest_Popularity'] >= 0.01]
    strong_signal_topics_df = strong_signal_topics_df[strong_signal_topics_df['Latest_Popularity'] >= 0.01]

    return noise_topics_df, weak_signal_topics_df, strong_signal_topics_df


########################################################################################################################



def save_signal_evolution_data(all_merge_histories_df: pd.DataFrame, topic_sizes: Dict[int, Dict[str, Any]], 
                               topic_last_popularity: Dict[int, float], topic_last_update: Dict[int, pd.Timestamp],
                               min_datetime: pd.Timestamp, max_datetime: pd.Timestamp, window_size: int, granularity: int,
                               start_timestamp: pd.Timestamp = None, end_timestamp: pd.Timestamp = None) -> None:
    """
    Save the history of noise, weak, and strong signal dataframes incrementally within the specified start and end timestamps.
    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
        topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information.
        topic_last_popularity (Dict[int, float]): Dictionary storing the last popularity of each topic.
        topic_last_update (Dict[int, pd.Timestamp]): Dictionary storing the last update timestamp of each topic.
        min_datetime (pd.Timestamp): The minimum timestamp to start from.
        max_datetime (pd.Timestamp): The maximum timestamp to end at.
        window_size (int): The size of the retrospective window in days.
        granularity (int): The granularity of the timestamps in days.
        start_timestamp (pd.Timestamp, optional): The start timestamp for saving the data. If not provided, defaults to min_datetime + window_size.
        end_timestamp (pd.Timestamp, optional): The end timestamp for saving the data. If not provided, defaults to max_datetime.
    """
    window_size_timedelta = pd.Timedelta(days=window_size)
    granularity_timedelta = pd.Timedelta(days=granularity)

    if start_timestamp is None:
        start_timestamp = min_datetime + window_size_timedelta
    if end_timestamp is None:
        end_timestamp = max_datetime

    # Determine the number of iterations for tqdm
    total_iterations = (end_timestamp - start_timestamp) // granularity_timedelta + 1

    save_path = "wattelse/bertopic/app/weak_signals/ablation_study/signal_evolution_data_arxiv"
    os.makedirs(save_path, exist_ok=True)

    q1_values = []
    q3_values = []
    timestamps_over_time = []

    for current_timestamp in tqdm(pd.date_range(start=start_timestamp, 
                                                end=end_timestamp, 
                                                freq=granularity_timedelta), total=total_iterations, desc='Processing timestamps'):
        window_start = current_timestamp - window_size_timedelta
        window_end = current_timestamp

        # Collect popularity values above 0.001 within the specified window
        all_popularity_values = [
            popularity for topic, data in topic_sizes.items()
            for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
            if window_start <= timestamp <= window_end and popularity > 0.01
        ]

        # Calculate the 10th and 50th percentiles of popularity values
        if all_popularity_values:
            q1 = np.percentile(all_popularity_values, 10)
            q3 = np.percentile(all_popularity_values, 90)
        else:
            q1, q3 = 0, 0

        q1_values.append(q1)
        q3_values.append(q3)

        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = classify_signals(topic_sizes, window_start, window_end, q1, q3, keep_documents=False)

        # Save the dataframes for each signal type incrementally
        noise_topics_df.to_pickle(os.path.join(save_path, f"noise_topics_df_{current_timestamp.strftime('%Y-%m-%d')}.pkl"))
        weak_signal_topics_df.to_pickle(os.path.join(save_path, f"weak_signal_topics_df_{current_timestamp.strftime('%Y-%m-%d')}.pkl"))
        strong_signal_topics_df.to_pickle(os.path.join(save_path, f"strong_signal_topics_df_{current_timestamp.strftime('%Y-%m-%d')}.pkl"))

        timestamps_over_time.append(current_timestamp)

    # Save the window size, granularity, timestamps, q1_values, and q3_values
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        metadata = {
            "window_size": window_size,
            "granularity": granularity,
            "timestamps": timestamps_over_time,
            "q1_values": q1_values,
            "q3_values": q3_values
        }
        pickle.dump(metadata, f)

    # Group the existing dataframes and delete the individual dataframes
    group_and_delete_dataframes(save_path)




    

def group_and_delete_dataframes(save_path: str) -> None:
    # Get a list of all files in the save_path directory
    all_files = os.listdir(save_path)

    # Filter files based on their prefix
    noise_files = sorted([file for file in all_files if file.startswith("noise_topics_df_")])
    weak_signal_files = sorted([file for file in all_files if file.startswith("weak_signal_topics_df_")])
    strong_signal_files = sorted([file for file in all_files if file.startswith("strong_signal_topics_df_")])

    # Load dataframes and store them in lists
    noise_df_list = [pd.read_pickle(os.path.join(save_path, file)) for file in noise_files]
    weak_signal_df_list = [pd.read_pickle(os.path.join(save_path, file)) for file in weak_signal_files]
    strong_signal_df_list = [pd.read_pickle(os.path.join(save_path, file)) for file in strong_signal_files]

    # Save the lists of dataframes
    with open(os.path.join(save_path, "noise_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(noise_df_list, f)
    with open(os.path.join(save_path, "weak_signal_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(weak_signal_df_list, f)
    with open(os.path.join(save_path, "strong_signal_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(strong_signal_df_list, f)

    # Delete the individual dataframes
    for file in noise_files + weak_signal_files + strong_signal_files:
        os.remove(os.path.join(save_path, file))