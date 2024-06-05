from typing import Dict, List, Tuple, Any
import pandas as pd
from bertopic import BERTopic
import numpy as np
import os
import pickle
from collections import defaultdict
import streamlit as st

def detect_weak_signals_zeroshot(topic_models: Dict[pd.Timestamp, BERTopic], zeroshot_topic_list: List[str]) -> Dict[str, Dict[pd.Timestamp, Dict[str, any]]]:
    """
    Detect weak signals based on the zero-shot list of topics to monitor.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): Dictionary of BERTopic models for each timestamp.
        zeroshot_topic_list (List[str]): List of topics to monitor for weak signals.

    Returns:
        Dict[str, Dict[pd.Timestamp, Dict[str, any]]]: Dictionary of weak signal trends for each monitored topic.
    """
    weak_signal_trends = {}

    for topic in zeroshot_topic_list:
        weak_signal_trends[topic] = {}

        for timestamp, topic_model in topic_models.items():
            topic_info = topic_model.topic_info_df

            for _, row in topic_info.iterrows():
                if row['Name'] == topic:
                    weak_signal_trends[topic][timestamp] = {
                        'Representation': row['Representation'],
                        'Representative_Docs': row['Representative_Docs'],
                        'Count': row['Count'],
                        'Document_Count': row['Document_Count']
                    }
                    break

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
                topic_sizes[current_topic]['Representations'].append(f"{current_timestamp}: {'_'.join(row['Representation1'])}")
                topic_sizes[current_topic]['Documents'].extend(row['Documents1'])
                topic_sizes[current_topic]['Sources'].extend(row['Source1'])
                topic_sizes[current_topic]['Docs_Count'] = row['Document_Count1']
                topic_sizes[current_topic]['Paragraphs_Count'] = row['Count1']

                # Update the topic's last popularity and last update timestamp
                topic_last_popularity[current_topic] = row['Document_Count1']
                topic_last_update[current_topic] = current_timestamp
            
            # Update the topic's data with the new timestamp and aggregated popularity
            next_timestamp = current_timestamp + granularity_timedelta
            
            topic_sizes[current_topic]['Timestamps'].append(next_timestamp)
            topic_sizes[current_topic]['Popularity'].append(topic_last_popularity[current_topic] + row['Document_Count2'])
            topic_sizes[current_topic]['Representations'].append(f"{next_timestamp}: {'_'.join(row['Representation2'])}")
            topic_sizes[current_topic]['Documents'].extend(row['Documents2'])
            topic_sizes[current_topic]['Sources'].extend(row['Source2'])
            topic_sizes[current_topic]['Docs_Count'] += row['Document_Count2']
            topic_sizes[current_topic]['Paragraphs_Count'] += row['Count2']

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
            topic_sizes[topic]['Representations'].append(f"{current_timestamp}: {'_'.join(topic_sizes[topic]['Representations'][-1].split(': ')[1:])}")
            
            topic_last_popularity[topic] = decayed_popularity

    return topic_sizes, topic_last_popularity, topic_last_update


########################################################################################################################

@st.cache_data
def classify_signals(topic_sizes: Dict[int, Dict[str, Any]], window_start: pd.Timestamp, window_end: pd.Timestamp, q1: float, q3: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify signals into weak signal and strong signal dataframes.
    
    Args:
        topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information.
        window_start (pd.Timestamp): The start timestamp of the window.
        window_end (pd.Timestamp): The end timestamp of the window.
        q1 (float): The 10th percentile of popularity values.
        q3 (float): The 50th percentile of popularity values.
    
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
        window_popularities = [
            (timestamp, popularity) for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
            if window_start <= timestamp <= window_end and popularity > 0.001
        ]
        if window_popularities:
            latest_timestamp, latest_popularity = window_popularities[-1]
            if latest_popularity < q1:
                noise_topics.append((topic, latest_popularity, latest_timestamp))
            elif q1 <= latest_popularity <= q3:
                weak_signal_topics.append((topic, latest_popularity, latest_timestamp))
            else:
                strong_signal_topics.append((topic, latest_popularity, latest_timestamp))

    # Create DataFrames for each category using list comprehensions
    noise_topics_data = [{
        'Topic': topic,
        'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
        'Latest Popularity': latest_popularity,
        'Docs_Count': topic_sizes[topic]['Docs_Count'],
        'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
        'Latest_Timestamp': latest_timestamp,
        'Representations': topic_sizes[topic]['Representations'],
        'Documents': topic_sizes[topic]['Documents'],
        'Sources': list(set(topic_sizes[topic]['Sources'])),
        'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
    } for topic, latest_popularity, latest_timestamp in noise_topics]

    weak_signal_topics_data = [{
        'Topic': topic,
        'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
        'Latest Popularity': latest_popularity,
        'Docs_Count': topic_sizes[topic]['Docs_Count'],
        'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
        'Latest_Timestamp': latest_timestamp,
        'Representations': topic_sizes[topic]['Representations'],
        'Documents': topic_sizes[topic]['Documents'],
        'Sources': list(set(topic_sizes[topic]['Sources'])),
        'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
    } for topic, latest_popularity, latest_timestamp in weak_signal_topics]

    strong_signal_topics_data = [{
        'Topic': topic,
        'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
        'Latest Popularity': latest_popularity,
        'Docs_Count': topic_sizes[topic]['Docs_Count'],
        'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
        'Latest_Timestamp': latest_timestamp,
        'Representations': topic_sizes[topic]['Representations'],
        'Documents': topic_sizes[topic]['Documents'],
        'Sources': list(set(topic_sizes[topic]['Sources'])),
        'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
    } for topic, latest_popularity, latest_timestamp in strong_signal_topics]

    # Create DataFrames for each category
    noise_topics_df = pd.DataFrame(noise_topics_data)
    weak_signal_topics_df = pd.DataFrame(weak_signal_topics_data)
    strong_signal_topics_df = pd.DataFrame(strong_signal_topics_data)

    return noise_topics_df, weak_signal_topics_df, strong_signal_topics_df


########################################################################################################################


def save_signal_evolution_data(all_merge_histories_df: pd.DataFrame, topic_sizes: Dict[int, Dict[str, Any]], 
                               topic_last_popularity: Dict[int, float], topic_last_update: Dict[int, pd.Timestamp],
                               min_datetime: pd.Timestamp, max_datetime: pd.Timestamp, window_size: int, granularity: int) -> None:
    """
    Save the entire history of noise, weak, and strong signal dataframes.
    
    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
        topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information.
        topic_last_popularity (Dict[int, float]): Dictionary storing the last popularity of each topic.
        topic_last_update (Dict[int, pd.Timestamp]): Dictionary storing the last update timestamp of each topic.
        min_datetime (pd.Timestamp): The minimum timestamp to start from.
        max_datetime (pd.Timestamp): The maximum timestamp to end at.
        window_size (int): The size of the retrospective window in days.
        granularity (int): The granularity of the timestamps in days.
    """
    window_size_timedelta = pd.Timedelta(days=window_size)
    granularity_timedelta = pd.Timedelta(days=granularity)

    current_timestamp = min_datetime + window_size_timedelta
    end_timestamp = max_datetime

    noise_dfs_over_time = []
    weak_signal_dfs_over_time = []
    strong_signal_dfs_over_time = []
    timestamps_over_time = []
    q1_values = []
    q3_values = []

    while current_timestamp <= end_timestamp:
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
            q3 = np.percentile(all_popularity_values, 50)
        else:
            q1, q3 = 0, 0

        q1_values.append(q1)
        q3_values.append(q3)

        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = classify_signals(topic_sizes, window_start, window_end, q1, q3)

        # Append the dataframes for each signal type to the respective lists
        noise_dfs_over_time.append(noise_topics_df)
        weak_signal_dfs_over_time.append(weak_signal_topics_df)
        strong_signal_dfs_over_time.append(strong_signal_topics_df)
        timestamps_over_time.append(current_timestamp)

        # Move to the next timestamp
        current_timestamp += granularity_timedelta

    # Save the retrieved dataframes and related information
    save_path = "wattelse/bertopic/app/weak_signals/ablation_study/signal_evolution_data"
    os.makedirs(save_path, exist_ok=True)

    # Save the lists of dataframes for each signal type
    with open(os.path.join(save_path, "noise_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(noise_dfs_over_time, f)
    with open(os.path.join(save_path, "weak_signal_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(weak_signal_dfs_over_time, f)
    with open(os.path.join(save_path, "strong_signal_dfs_over_time.pkl"), "wb") as f:
        pickle.dump(strong_signal_dfs_over_time, f)

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