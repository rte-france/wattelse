import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Tuple 
from bertopic import BERTopic
import streamlit as st
import numpy as np
from loguru import logger
import pandas as pd
from weak_signals import classify_signals, save_signal_evolution_data
from plotly_resampler import FigureResampler, FigureWidgetResampler
import time
from plotly_resampler import register_plotly_resampler

from multiprocessing import Process
import streamlit.components.v1 as components

def plot_num_topics_and_outliers(topic_models: Dict[pd.Timestamp, BERTopic]) -> None:
    """
    Plot the number of topics detected and the size of the outlier topic for each model.
    
    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp and the value is the corresponding model.
    """
    num_topics = [len(model.get_topic_info()) for model in topic_models.values()]
    fig_num_topics = go.Figure(data=[go.Bar(x=list(topic_models.keys()), y=num_topics)])
    fig_num_topics.update_layout(title="Number of Topics Detected", xaxis_title="Time Period", yaxis_title="Number of Topics")
    st.plotly_chart(fig_num_topics, use_container_width=True)

    outlier_sizes = [model.get_topic_info().loc[model.get_topic_info()['Topic'] == -1]['Count'].values[0] if -1 in model.get_topic_info()['Topic'].values else 0 for model in topic_models.values()]
    fig_outlier_sizes = go.Figure(data=[go.Bar(x=list(topic_models.keys()), y=outlier_sizes)])
    fig_outlier_sizes.update_layout(title="Size of Outlier Topic", xaxis_title="Time Period", yaxis_title="Size of Outlier Topic")
    st.plotly_chart(fig_outlier_sizes, use_container_width=True)

########################################################################################################################


def prepare_source_topic_data(doc_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for plotting topics per source by counting the number of unique documents for each source and topic combination.
    
    Args:
        doc_info_df (pd.DataFrame): The document information DataFrame containing 'source', 'Topic', 'document_id', and 'Representation' columns.
    
    Returns:
        pd.DataFrame: A DataFrame with 'source', 'Topic', 'Count', and 'Representation' columns.
    """
    source_topic_counts = doc_info_df.groupby(['source', 'Topic'])['document_id'].nunique().reset_index(name='Count')
    topic_representations = doc_info_df.groupby('Topic')['Representation'].first().to_dict()
    source_topic_counts['Representation'] = source_topic_counts['Topic'].map(topic_representations)
    source_topic_counts = source_topic_counts.sort_values(['source', 'Count'], ascending=[True, False])
    return source_topic_counts

########################################################################################################################

def plot_topics_per_timestamp(topic_models: Dict[pd.Timestamp, BERTopic]) -> None:
    """
    Plot the topics discussed per source for each timestamp.
    
    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp and the value is the corresponding model.
    """
    with st.expander("Explore topic models") : 
        model_periods = sorted(topic_models.keys())
        selected_model_period = st.select_slider("Select Model", options=model_periods, key='model_slider')
        selected_model = topic_models[selected_model_period]
        
        source_topic_counts = prepare_source_topic_data(selected_model.doc_info_df)
        
        fig = go.Figure()
        
        for topic, topic_data in source_topic_counts.groupby('Topic'):
            fig.add_trace(go.Bar(
                x=topic_data['source'],
                y=topic_data['Count'],
                name=str(topic)+'_'+'_'.join(topic_data['Representation'].iloc[0][:5]),
                hovertemplate='Source: %{x}<br>Topic: %{customdata}<br>Number of documents: %{y}<extra></extra>',
                customdata= topic_data['Representation']
            ))
        
        fig.update_layout(
            title='Talked About Topics per Source',
            xaxis_title='Source',
            yaxis_title='Number of Paragraphs',
            barmode='stack',
            legend_title='Topics'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(selected_model.doc_info_df[['Paragraph', 'document_id', 'Topic', 'Representation', 'source']], use_container_width=True)
        st.dataframe(selected_model.topic_info_df, use_container_width=True)

########################################################################################################################

@st.cache_data
def create_topic_size_evolution_figure(topic_ids=None):
    fig = go.Figure()

    topic_sizes = st.session_state.topic_sizes

    if topic_ids is None:
        # If topic_ids is not provided, include all topics
        sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[0])
    else:
        # If topic_ids is provided, filter the topics based on the specified IDs
        sorted_topics = [(topic_id, topic_sizes[topic_id]) for topic_id in topic_ids if topic_id in topic_sizes]

    # Create traces for each selected topic
    for topic, data in sorted_topics:
        fig.add_trace(go.Scattergl(
            x=data['Timestamps'],
            y=data['Popularity'],
            mode='lines+markers',
            name=f"Topic {topic} : {data['Representations'][0].split('_')[:5]}",
            hovertemplate='Topic: %{text}<br>Timestamp: %{x}<br>Popularity: %{y}<br>Representation: %{customdata}<extra></extra>',
            text=[f"Topic {topic}"] * len(data['Timestamps']),
            customdata=[rep for rep in data['Representations']],
        ))

    fig.update_layout(
        title="Signal Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Popularity",
        hovermode="closest"
    )

    return fig


def create_topic_size_evolution_figure_labeled(topic_labels):
    fig = go.Figure()

    topic_sizes = st.session_state.topic_sizes

    weak_signal_color = 'rgba(255, 0, 0, 0.8)'  # Intense red color for weak signals
    strong_signal_color = 'rgba(0, 255, 0, 0.3)'  # Light green color for strong signals

    for topic_label in topic_labels:
        topic_id = int(topic_label[:-1])
        label = topic_label[-1].lower()

        if topic_id in topic_sizes:
            data = topic_sizes[topic_id]
            timestamps = data['Timestamps']
            popularity = data['Popularity']
            representations = data['Representations']

            if label == 'w':
                trace_color = weak_signal_color
                trace_name = f"Weak Signal - Topic {topic_id}"
            elif label == 's':
                trace_color = strong_signal_color
                trace_name = f"Strong Signal - Topic {topic_id}"
            else:
                continue

            fig.add_trace(go.Scattergl(
                x=timestamps,
                y=popularity,
                mode='lines+markers',
                name=f"Topic {topic_id} : {data['Representations'][0].split('_')[:5]}",
                hovertemplate='Topic: %{text}<br>Timestamp: %{x}<br>Popularity: %{y}<br>Representation: %{customdata}<extra></extra>',
                text=[f"Topic {topic_id}"] * len(timestamps),
                customdata=[rep for rep in representations],
                line=dict(color=trace_color),
                marker=dict(color=trace_color)
            ))

    fig.update_layout(
        title="Signal Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Popularity",
        hovermode="closest"
    )

    return fig


def plot_topic_size_evolution(fig, window_size: int, granularity: int, current_date, min_datetime, max_datetime) -> Tuple[float, float]:
    """
    Plot the evolution of topic sizes over time.
    
    Args:
        fig (FigureWidgetResampler): The cached figure to plot.
        window_size (int): The retrospective window size in days.
        granularity (int): The granularity of the timestamps in days.
        current_date (datetime): The current date selected by the user.
        min_datetime (datetime): The minimum datetime value.
        max_datetime (datetime): The maximum datetime value.
    
    Returns:
        Tuple[float, float]: The q1 and q3 values representing the 10th and 50th percentiles of popularity values.
    """
    with st.expander("Topic Popularity Evolution", expanded=True):
        topic_sizes = st.session_state.topic_sizes
        topic_last_popularity = st.session_state.topic_last_popularity
        topic_last_update = st.session_state.topic_last_update

        window_size_timedelta = pd.Timedelta(days=window_size)

        window_end = pd.to_datetime(current_date)
        window_start = window_end - window_size_timedelta

        # Collect popularity values above 0.01 within the specified window
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

        # st.write(window_end)
        st.write(f"Noise Threshold : {q1}")
        st.write(f"Strong Signal Threshold : {q3}")

        st.plotly_chart(fig, use_container_width=True)


        # Update the figure layout to limit the display range to the window
        fig.update_layout(
            title='Popularity Evolution',
            xaxis_title='Timestamp',
            yaxis_title='Popularity',
            hovermode='closest',
            xaxis_range=[window_start, window_end]  # Set the range of the x-axis to the retrospective window
        )

        # Define the columns to display in the DataFrames
        columns = ['Topic', 'Sources', 'Source_Diversity', 'Representation', 'Latest_Popularity', 'Docs_Count', 'Paragraphs_Count', 'Latest_Timestamp', 'Documents']

        # Classify topics based on their popularity behavior at the final timestamp
        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = classify_signals(topic_sizes, window_start, window_end, q1, q3)

        # Display the DataFrames for each category
        st.subheader(":grey[Noise]")
        if not noise_topics_df.empty:
            st.dataframe(noise_topics_df[columns].sort_values(by=['Topic', 'Latest_Popularity'], ascending=[False, False]))
        else:
            st.info(f"No noisy signals were detected at timestamp {window_end}.")

        st.subheader(":orange[Weak Signals]")
        if not weak_signal_topics_df.empty:
            st.dataframe(weak_signal_topics_df[columns].sort_values(by=['Latest_Popularity'], ascending=True))
        else:
            st.info(f"No weak signals were detected at timestamp {window_end}.")

        st.subheader(":green[Strong Signals]")
        if not strong_signal_topics_df.empty:
            st.dataframe(strong_signal_topics_df[columns].sort_values(by=['Topic', 'Latest_Popularity'], ascending=[False, False]))
        else:
            st.info(f"No strong signals were detected at timestamp {window_end}.")


        # Add date range picker for saving signal evolution data
        start_date, end_date = st.date_input("Select date range for saving signal evolution data:", value=(min_datetime.date(), max_datetime.date()), min_value=min_datetime.date(), max_value=max_datetime.date())

        if st.button("Save Signal Evolution Data"):
            save_signal_evolution_data(st.session_state.all_merge_histories_df, dict(topic_sizes), topic_last_popularity, topic_last_update, min_datetime, max_datetime, window_size, granularity, pd.Timestamp(start_date), pd.Timestamp(end_date))
            st.success("Signal evolution data saved successfully!")

        return window_start, window_end

########################################################################################################################



# def plot_topic_size_evolution(all_merge_histories_df: pd.DataFrame, granularity: int) -> Tuple[float, float]:
#     """
#     Plot the evolution of topic sizes over time.
    
#     Args:
#         all_merge_histories_df (pd.DataFrame): The DataFrame containing the merge histories of topics.
#         granularity (int): The granularity of the timestamps in days.
    
#     Returns:
#         Tuple[float, float]: The q1 and q3 values representing the 10th and 50th percentiles of popularity values.
#     """
#     with st.expander("Topic Popularity Evolution", expanded=True):
#         fig = go.Figure()
#         topic_sizes = defaultdict(lambda: defaultdict(list))
#         topic_last_popularity = {}
#         topic_last_update = {}

#         min_timestamp = all_merge_histories_df['Timestamp'].min()
#         max_timestamp = all_merge_histories_df['Timestamp'].max()

#         window_size = st.number_input("Retrospective Period (days)", min_value=1, max_value=365, value=10, key='window_size')
#         window_size_timedelta = pd.Timedelta(days=window_size)

#         min_datetime = min_timestamp.to_pydatetime()
#         max_datetime = max_timestamp.to_pydatetime()

#         granularity_timedelta = pd.Timedelta(days=granularity)

#         window_end = st.slider(
#             "Current date",
#             min_value=min_datetime + window_size_timedelta,
#             max_value=max_datetime,
#             value=min_datetime + window_size_timedelta,
#             step=granularity_timedelta,
#             format="YYYY-MM-DD",
#         )
        
#         window_start = window_end - window_size_timedelta

#         # Create a range of timestamps from min_datetime to max_datetime with the specified granularity
#         timestamps = pd.date_range(start=min_datetime, end=max_datetime, freq=granularity_timedelta)

#         # Iterate over each timestamp
#         for current_timestamp in timestamps:
#             # Filter the merge history DataFrame for the current timestamp
#             current_df = all_merge_histories_df[all_merge_histories_df['Timestamp'] == current_timestamp]
            
#             # Iterate over each topic at the current timestamp
#             for _, row in current_df.iterrows():
#                 current_topic = row['Topic1']

#                 if current_topic not in topic_sizes:
#                     # Initialize the topic's data with the first point corresponding to Timestamp1 and popularity Document_Count1
#                     topic_sizes[current_topic]['Timestamps'].append(current_timestamp)
#                     topic_sizes[current_topic]['Popularity'].append(row['Document_Count1'])
#                     topic_sizes[current_topic]['Representations'].append(f"{current_timestamp}: {'_'.join(row['Representation1'])}")
#                     topic_sizes[current_topic]['Documents'].extend(row['Documents1'])
#                     topic_sizes[current_topic]['Sources'].extend(row['Source1'])
#                     topic_sizes[current_topic]['Docs_Count'] = row['Document_Count1']
#                     topic_sizes[current_topic]['Paragraphs_Count'] = row['Count1']

#                     # Update the topic's last popularity and last update timestamp
#                     topic_last_popularity[current_topic] = row['Document_Count1']
#                     topic_last_update[current_topic] = current_timestamp
#                 else:
#                     # Retrieve the last logged popularity value
#                     last_popularity = topic_last_popularity[current_topic]
#                     last_update = topic_last_update[current_topic]

#                     # Calculate the number of granularities since the last update
#                     time_diff = current_timestamp - last_update
#                     periods_since_last_update = time_diff // granularity_timedelta

#                     # Apply exponential decay to the last popularity based on the number of periods since the last update
#                     decayed_popularity = last_popularity * np.exp(-0.01 * (periods_since_last_update)**2)

#                     # Append the decayed popularity to the topic's data
#                     topic_sizes[current_topic]['Timestamps'].append(current_timestamp)
#                     topic_sizes[current_topic]['Popularity'].append(decayed_popularity)
#                     topic_sizes[current_topic]['Representations'].append(f"{current_timestamp}: {'_'.join(topic_sizes[current_topic]['Representations'][-1].split(': ')[1:])}")

#                     # Update the topic's last popularity
#                     topic_last_popularity[current_topic] = decayed_popularity

#             # Iterate over each topic at the current timestamp
#             for _, row in current_df.iterrows():
#                 current_topic = row['Topic1']

#                 # Check if the topic at this timestamp got merged with multiple topics from the next timestamp
#                 next_timestamp = row['Timestamp'] + granularity_timedelta
                
#                 # Update the topic's data with the new timestamp and aggregated popularity
#                 topic_sizes[current_topic]['Timestamps'].append(next_timestamp)
#                 topic_sizes[current_topic]['Popularity'].append(topic_last_popularity[current_topic] + row['Document_Count2'])
#                 topic_sizes[current_topic]['Representations'].append(f"{next_timestamp}: {'_'.join(row['Representation2'])}")
#                 topic_sizes[current_topic]['Documents'].extend(row['Documents2'])
#                 topic_sizes[current_topic]['Sources'].extend(row['Source2'])
#                 topic_sizes[current_topic]['Docs_Count'] += row['Document_Count2']
#                 topic_sizes[current_topic]['Paragraphs_Count'] += row['Count2']

#                 # Update the topic's last popularity and last update timestamp
#                 topic_last_popularity[current_topic] = topic_last_popularity[current_topic] + row['Document_Count2']
#                 topic_last_update[current_topic] = next_timestamp

#         # Sort the topics based on their topic ID
#         sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[0])

#         # Create a Plotly figure and add traces for each topic
#         for topic, data in sorted_topics:
#             fig.add_trace(go.Scatter(
#                 x=data['Timestamps'],
#                 y=data['Popularity'],
#                 mode='lines+markers',
#                 name=f"Topic {topic} : {data['Representations'][-1].split(': ')[1].split('_')[:5]}",
#                 hovertemplate='Topic: %{text}<br>Timestamp: %{x}<br>Popularity: %{y}<br>Representation: %{customdata}<extra></extra>',
#                 text=[f"Topic {topic}"] * len(data['Timestamps']),
#                 customdata=[rep.split(': ')[1] for rep in data['Representations']],
#                 line_shape='spline'
#             ))

#         # Collect popularity values above 0.001 within the specified window
#         all_popularity_values = [
#             popularity for topic, data in sorted_topics
#             for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
#             if window_start <= timestamp <= window_end and popularity > 0.01
#         ]

#         # Calculate the 10th and 50th percentiles of popularity values
#         if all_popularity_values:
#             q1 = np.percentile(all_popularity_values, 10)
#             q3 = np.percentile(all_popularity_values, 50)
#         else:
#             q1, q3 = 0, 0

#         st.write(f"Noise Threshold : {q1}")
#         st.write(f"Strong Signal Threshold : {q3}")

#         # Add horizontal lines for the 10th and 50th percentiles
#         fig.add_shape(type="line", x0=window_start, y0=q1, x1=window_end, y1=q1, line=dict(color="red", width=2, dash="dash"))
#         fig.add_shape(type="line", x0=window_start, y0=q3, x1=window_end, y1=q3, line=dict(color="green", width=2, dash="dash"))

#         # Update the figure layout to limit the display range to the window
#         fig.update_layout(
#             title='Popularity Evolution',
#             xaxis_title='Timestamp',
#             yaxis_title='Popularity',
#             hovermode='closest',
#             xaxis_range=[window_start, window_end]  # Set the range of the x-axis to the retrospective window
#         )

#         # Display the plot using Streamlit
#         st.plotly_chart(fig, use_container_width=True)

#         # # Call the save_signal_evolution_data function to save the dataframes and metadata
#         save_signal_evolution_data(all_merge_histories_df, dict(topic_sizes), topic_last_popularity, topic_last_update, 
#                                    min_datetime, max_datetime, window_size, granularity)

#         # Define the columns to display in the DataFrames
#         columns = ['Topic', 'Sources', 'Source_Diversity', 'Representation', 'Latest_Popularity', 'Docs_Count', 'Paragraphs_Count', 'Latest_Timestamp', 'Documents']

#         # Classify topics based on their popularity behavior at the final timestamp
#         noise_topics = []
#         weak_signal_topics = []
#         strong_signal_topics = []

#         for topic, data in sorted_topics:
#             window_popularities = [
#                 (timestamp, popularity) for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
#                 if window_start <= timestamp <= window_end and popularity > 0.0001
#             ]
#             if window_popularities:
#                 latest_timestamp, latest_popularity = window_popularities[-1]
#                 if latest_popularity < q1:
#                     noise_topics.append((topic, latest_popularity, latest_timestamp))
#                 elif q1 <= latest_popularity <= q3:
#                     weak_signal_topics.append((topic, latest_popularity, latest_timestamp))
#                 else:
#                     strong_signal_topics.append((topic, latest_popularity, latest_timestamp))

#         # Create DataFrames for each category using list comprehensions
#         noise_topics_data = [{
#             'Topic': topic,
#             'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
#             'Latest_Popularity': latest_popularity,
#             'Docs_Count': topic_sizes[topic]['Docs_Count'],
#             'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
#             'Latest_Timestamp': latest_timestamp,
#             'Representations': topic_sizes[topic]['Representations'],
#             'Documents': topic_sizes[topic]['Documents'],
#             'Sources': list(set(topic_sizes[topic]['Sources'])),
#             'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
#         } for topic, latest_popularity, latest_timestamp in noise_topics]

#         weak_signal_topics_data = [{
#             'Topic': topic,
#             'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
#             'Latest_Popularity': latest_popularity,
#             'Docs_Count': topic_sizes[topic]['Docs_Count'],
#             'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
#             'Latest_Timestamp': latest_timestamp,
#             'Representations': topic_sizes[topic]['Representations'],
#             'Documents': topic_sizes[topic]['Documents'],
#             'Sources': list(set(topic_sizes[topic]['Sources'])),
#             'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
#         } for topic, latest_popularity, latest_timestamp in weak_signal_topics]

#         strong_signal_topics_data = [{
#             'Topic': topic,
#             'Representation': topic_sizes[topic]['Representations'][-1].split(': ')[1],
#             'Latest_Popularity': latest_popularity,
#             'Docs_Count': topic_sizes[topic]['Docs_Count'],
#             'Paragraphs_Count': topic_sizes[topic]['Paragraphs_Count'],
#             'Latest_Timestamp': latest_timestamp,
#             'Representations': topic_sizes[topic]['Representations'],
#             'Documents': topic_sizes[topic]['Documents'],
#             'Sources': list(set(topic_sizes[topic]['Sources'])),
#             'Source_Diversity': len(set(topic_sizes[topic]['Sources']))
#         } for topic, latest_popularity, latest_timestamp in strong_signal_topics]

#         # Create DataFrames for each category
#         noise_topics_df = pd.DataFrame(noise_topics_data)
#         weak_signal_topics_df = pd.DataFrame(weak_signal_topics_data)
#         strong_signal_topics_df = pd.DataFrame(strong_signal_topics_data)

#         # Display the DataFrames for each category
#         st.subheader(":grey[Noise]")
#         if not noise_topics_df.empty:
#             st.dataframe(noise_topics_df[columns].sort_values(by=['Topic', 'Latest_Popularity'], ascending=[False, False]))
#         else:
#             st.info(f"No noisy signals were detected at timestamp {window_end}.")

#         st.subheader(":orange[Weak Signals]")
#         if not weak_signal_topics_df.empty:
#             st.dataframe(weak_signal_topics_df[columns].sort_values(by=['Latest_Popularity'], ascending=True))
#         else:
#             st.info(f"No weak signals were detected at timestamp {window_end}.")

#         st.subheader(":green[Strong Signals]")
#         if not strong_signal_topics_df.empty:
#             st.dataframe(strong_signal_topics_df[columns].sort_values(by=['Topic', 'Latest_Popularity'], ascending=[False, False]))
#         else:
#             st.info(f"No strong signals were detected at timestamp {window_end}.")

#         return window_start, window_end





# def save_signal_evolution_data(all_merge_histories_df, topic_sizes, topic_last_popularity, topic_last_update, min_datetime, max_datetime, window_size, granularity):
#     """
#     Save the entire history of noise, weak, and strong signal dataframes from the beginning to the end.
    
#     Args:
#         all_merge_histories_df (pd.DataFrame): The DataFrame containing the merge histories of topics.
#         topic_sizes (dict): Dictionary storing topic sizes and related information.
#         topic_last_popularity (dict): Dictionary storing the last popularity of each topic.
#         topic_last_update (dict): Dictionary storing the last update timestamp of each topic.
#         min_datetime (datetime): The minimum timestamp to start from.
#         max_datetime (datetime): The maximum timestamp to end at.
#         window_size (int): The size of the retrospective window in days.
#         granularity (int): The granularity of the timestamps in days.

#     """
#     window_size_timedelta = pd.Timedelta(days=window_size)
#     granularity_timedelta = pd.Timedelta(days=granularity)

#     current_timestamp = min_datetime + window_size_timedelta
#     end_timestamp = max_datetime

#     noise_dfs_over_time = []
#     weak_signal_dfs_over_time = []
#     strong_signal_dfs_over_time = []
#     timestamps_over_time = []
#     q1_values = []
#     q3_values = []

#     while current_timestamp <= end_timestamp:
#         window_start = current_timestamp - window_size_timedelta
#         window_end = current_timestamp

#         # Collect popularity values above 0.001 within the specified window
#         all_popularity_values = [
#             popularity for topic, data in topic_sizes.items()
#             for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
#             if window_start <= timestamp <= window_end and popularity > 0.01
#         ]

#         # Calculate the 10th and 50th percentiles of popularity values
#         if all_popularity_values:
#             q1 = np.percentile(all_popularity_values, 10)
#             q3 = np.percentile(all_popularity_values, 50)
#         else:
#             q1, q3 = 0, 0

#         q1_values.append(q1)
#         q3_values.append(q3)

#         noise_topics = []
#         weak_signal_topics = []
#         strong_signal_topics = []

#         for topic, data in topic_sizes.items():
#             window_popularities = [
#                 (timestamp, popularity) for timestamp, popularity in zip(data['Timestamps'], data['Popularity'])
#                 if window_start <= timestamp <= window_end and popularity > 0.01
#             ]
#             if window_popularities:
#                 latest_timestamp, latest_popularity = window_popularities[-1]
#                 if latest_popularity < q1:
#                     noise_topics.append((topic, latest_popularity, latest_timestamp))
#                 elif q1 <= latest_popularity <= q3:
#                     weak_signal_topics.append((topic, latest_popularity, latest_timestamp))
#                 else:
#                     strong_signal_topics.append((topic, latest_popularity, latest_timestamp))

#         # Create DataFrames for each category
#         noise_topics_data = []
#         weak_signal_topics_data = []
#         strong_signal_topics_data = []

#         for topic, latest_popularity, latest_timestamp in noise_topics:
#             topic_data = topic_sizes[topic]
#             noise_topics_data.append({
#                 'Topic': topic,
#                 'Representation': topic_data['Representations'][-1].split(': ')[1],
#                 'Latest_Popularity': latest_popularity,
#                 'Docs_Count': topic_data['Docs_Count'],
#                 'Paragraphs_Count': topic_data['Paragraphs_Count'],
#                 'Latest_Timestamp': latest_timestamp,
#                 'Representations': topic_data['Representations'],
#                 'Documents': topic_data['Documents'],
#                 'Sources': list(set(topic_data['Sources'])),
#                 'Source_Diversity': len(set(topic_data['Sources']))
#             })

#         for topic, latest_popularity, latest_timestamp in weak_signal_topics:
#             topic_data = topic_sizes[topic]
#             weak_signal_topics_data.append({
#                 'Topic': topic,
#                 'Representation': topic_data['Representations'][-1].split(': ')[1],
#                 'Latest_Popularity': latest_popularity,
#                 'Docs_Count': topic_data['Docs_Count'],
#                 'Paragraphs_Count': topic_data['Paragraphs_Count'],
#                 'Latest_Timestamp': latest_timestamp,
#                 'Representations': topic_data['Representations'],
#                 'Documents': topic_data['Documents'],
#                 'Sources': list(set(topic_data['Sources'])),
#                 'Source_Diversity': len(set(topic_data['Sources']))
#             })

#         for topic, latest_popularity, latest_timestamp in strong_signal_topics:
#             topic_data = topic_sizes[topic]
#             strong_signal_topics_data.append({
#                 'Topic': topic,
#                 'Representation': topic_data['Representations'][-1].split(': ')[1],
#                 'Latest_Popularity': latest_popularity,
#                 'Docs_Count': topic_data['Docs_Count'],
#                 'Paragraphs_Count': topic_data['Paragraphs_Count'],
#                 'Latest_Timestamp': latest_timestamp,
#                 'Representations': topic_data['Representations'],
#                 'Documents': topic_data['Documents'],
#                 'Sources': list(set(topic_data['Sources'])),
#                 'Source_Diversity': len(set(topic_data['Sources']))
#             })

#         # Create DataFrames for each category
#         noise_topics_df = pd.DataFrame(noise_topics_data)
#         weak_signal_topics_df = pd.DataFrame(weak_signal_topics_data)
#         strong_signal_topics_df = pd.DataFrame(strong_signal_topics_data)

#         # Append the dataframes for each signal type to the respective lists
#         noise_dfs_over_time.append(noise_topics_df)
#         weak_signal_dfs_over_time.append(weak_signal_topics_df)
#         strong_signal_dfs_over_time.append(strong_signal_topics_df)
#         timestamps_over_time.append(current_timestamp)

#         # Move to the next timestamp
#         current_timestamp += granularity_timedelta

#     # Save the retrieved dataframes and related information
#     save_path = "wattelse/bertopic/app/weak_signals/ablation_study/signal_evolution_data"
#     os.makedirs(save_path, exist_ok=True)

#     # Save the lists of dataframes for each signal type
#     with open(os.path.join(save_path, "noise_dfs_over_time.pkl"), "wb") as f:
#         pickle.dump(noise_dfs_over_time, f)
#     with open(os.path.join(save_path, "weak_signal_dfs_over_time.pkl"), "wb") as f:
#         pickle.dump(weak_signal_dfs_over_time, f)
#     with open(os.path.join(save_path, "strong_signal_dfs_over_time.pkl"), "wb") as f:
#         pickle.dump(strong_signal_dfs_over_time, f)

#     # Save the window size, granularity, timestamps, q1_values, and q3_values
#     with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
#         metadata = {
#             "window_size": window_size,
#             "granularity": granularity,
#             "timestamps": timestamps_over_time,
#             "q1_values": q1_values,
#             "q3_values": q3_values
#         }
#         pickle.dump(metadata, f)


def plot_newly_emerged_topics(all_new_topics_df: pd.DataFrame) -> None:
    """
    Plot the newly emerged topics over time.
    
    Args:
        all_new_topics_df (pd.DataFrame): The DataFrame containing information about newly emerged topics.
    """
    fig_new_topics = go.Figure()

    for timestamp, topics_df in all_new_topics_df.groupby('Timestamp'):
        fig_new_topics.add_trace(go.Scatter(
            x=[timestamp] * len(topics_df),
            y=topics_df['Document_Count'],
            text=topics_df['Topic'],
            mode='markers',
            marker=dict(
                size=topics_df['Document_Count'],
                sizemode='area',
                sizeref=2. * max(topics_df['Count']) / (40. ** 2),
                sizemin=4
            ),
            hovertemplate=(
                'Timestamp: %{x}<br>'
                'Topic ID: %{text}<br>'
                'Count: %{y}<br>'
                'Representation: %{customdata}<extra></extra>'
            ),
            customdata=topics_df['Representation']
        ))

    fig_new_topics.update_layout(
        title='Newly Emerged Topics',
        xaxis_title='Timestamp',
        yaxis_title='Topic Size',
        showlegend=False
    )

    with st.expander("Newly Emerged Topics", expanded=False):
        st.dataframe(all_new_topics_df[['Topic', 'Count', 'Document_Count', 'Representation', 'Documents', 'Timestamp']].sort_values(by=['Timestamp', 'Document_Count'], ascending=[True, False]))
        st.plotly_chart(fig_new_topics, use_container_width=True)

########################################################################################################################

def transform_merge_histories_for_sankey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the merge histories DataFrame to prepare it for creating a Sankey diagram.

    The function performs the following steps:
    1. Creates a 'Timestamp_Index' column that maps each timestamp to an index.
    2. Groups by 'Topic1' and collects the list of timestamp indices where each 'Topic1' value appears.
    3. Initializes variables to store the source, destination, representation, timestamp, and count values.
    4. Initializes dictionaries to store the mapping of (topic1, timestamp_index) to the new destination topic and the merged count.
    5. Groups by 'Timestamp' and processes each row to generate the source and destination topics, update the merged count, and determine the timestamp for the current row.
    6. Creates a new DataFrame with the transformed data, including source, destination, representation, timestamp, and count values.
    7. Converts lists to tuples in the 'Representation' column.
    8. Groups by 'Timestamp', 'Source', and 'Destination', and keeps the row with the smallest 'Count' value for each group.

    Args:
        df (pd.DataFrame): The merge histories DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame suitable for creating a Sankey diagram.
    """
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
            doc_count1 = row['Document_Count1']
            doc_count2 = row['Document_Count2']
            
            # Generate the source values for Topic1 and Topic2
            src_topic1 = f"{timestamp_index}_1_{topic1}"
            src_topic2 = f"{timestamp_index}_2_{topic2}"
            
            # Check if (topic1, timestamp_index) has a destination topic in the topic1_dest_map
            if (topic1, timestamp_index) in topic1_dest_map:
                dest_topic = topic1_dest_map[(topic1, timestamp_index)]
                
                # Update the merged count for the destination topic
                topic1_count_map[(topic1, timestamp_index)] += doc_count2
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
                topic1_count_map[(topic1, timestamp_index)] = doc_count1 + doc_count2
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
            count_values.extend([doc_count1, count_merged])

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

########################################################################################################################

def create_sankey_diagram_plotly(all_merge_histories_df: pd.DataFrame, search_term: str =None, max_pairs: int = None):
    """
    Create a Sankey diagram to visualize the topic merging process.
    
    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
        search_term (str): Optional search term to filter topics by keyword.
        max_pairs (int): Maximum number of topic pairs to display.
    
    Returns:
        go.Figure: The Plotly figure representing the Sankey diagram.
    """

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
        target_topic_id = target_node.split('_')
        if target_topic_id[-1] == 'new':
            target_topic_id = target_topic_id[-2]
        else:
            target_topic_id = target_topic_id[-1]

        
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
            customdata=[(link['timestamp'], link['source_topic_id'], link['target_topic_id'], link['representation'], link['value']) for link in links],
            hovertemplate='Timestamp: %{customdata[0]}<br />' +
              'Source Topic ID: %{customdata[1]}<br />' +
              'Target Topic ID: %{customdata[2]}<br />'
              'Representation: %{customdata[3]}<br />' +
              'Document Covered: %{customdata[4]}<extra></extra>',

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

########################################################################################################################

def create_sankey_diagram(all_merge_histories_df: pd.DataFrame) -> go.Figure:
    """
    Create a Sankey diagram to visualize the topic merging process.
    
    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
    
    Returns:
        go.Figure: The Plotly figure representing the Sankey diagram.
    """

    with st.expander("Topic Merging Process", expanded=False):

        # Create search box and slider using Streamlit
        search_term = st.text_input("Search topics by keyword:")
        max_pairs = st.slider("Max number of topic pairs to display", min_value=1, max_value=1000, value=20)

        # Transform the dataframe
        transformed_df = transform_merge_histories_for_sankey(all_merge_histories_df)
        
        # Create the Sankey diagram
        sankey_diagram = create_sankey_diagram_plotly(transformed_df, search_term, max_pairs)

        # Display the diagram using Streamlit in an expander
        st.plotly_chart(sankey_diagram, use_container_width=True)
    
    return sankey_diagram
