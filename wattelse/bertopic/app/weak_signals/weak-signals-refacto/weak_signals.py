from typing import Dict, List
import pandas as pd
from bertopic import BERTopic

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