import streamlit as st
import pandas as pd
import locale
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import umap
import numpy as np

from wattelse.bertopic.utils import TIMESTAMP_COLUMN, TEXT_COLUMN
from app_utils import plot_topics_over_time
from state_utils import restore_widget_state, register_widget, save_widget_state
from wattelse.bertopic.app.app_utils import plot_topics_over_time, compute_topics_over_time
from wattelse.bertopic.temporal_metrics_embedding import TempTopic

# Set locale for French date names
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Page configuration
st.set_page_config(page_title="Wattelse® topic", layout="wide")



# TempTopic output visualization functions
def plot_topic_evolution(temptopic, granularity, topics_to_show=None, n_neighbors=15, min_dist=0.1, metric='cosine', color_palette='Plotly'):
    topic_data = {}
    for topic_id in temptopic.final_df['Topic'].unique():
        topic_df = temptopic.final_df[temptopic.final_df['Topic'] == topic_id]
        
        # Parse timestamps based on granularity
        if granularity == "Week":
            timestamps = pd.to_datetime(topic_df['Timestamp'].apply(lambda x: f"{x}-1"), format="%Y-%W-%w")
        elif granularity == "Month":
            timestamps = pd.to_datetime(topic_df['Timestamp'], format="%Y-%m")
        elif granularity == "Year":
            timestamps = pd.to_datetime(topic_df['Timestamp'], format="%Y")
        else:  # Default to daily granularity
            timestamps = pd.to_datetime(topic_df['Timestamp'], format="%Y-%m-%d")

        topic_data[topic_id] = {
            'embeddings': topic_df['Embedding'].tolist(),
            'timestamps': timestamps,
            'words': topic_df['Words'].tolist()
        }

    all_embeddings = np.vstack([data['embeddings'] for data in topic_data.values()])
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, metric=metric, random_state=42)
    all_embeddings_umap = reducer.fit_transform(all_embeddings)

    start_idx = 0
    for topic_id, data in topic_data.items():
        end_idx = start_idx + len(data['embeddings'])
        data['embeddings_umap'] = all_embeddings_umap[start_idx:end_idx]
        start_idx = end_idx

    if topics_to_show is None or len(topics_to_show) == 0:
        topics_to_show = list(topic_data.keys())

    fig = go.Figure()

    for topic_id in topic_data.keys():
        data = topic_data[topic_id]
        visible = 'legendonly' if topic_id not in topics_to_show else True
        topic_words = ', '.join(data['words'][0].split(', ')[:3])  # Get first 3 words of the topic
        fig.add_trace(go.Scatter3d(
            x=data['embeddings_umap'][:, 0],
            y=data['embeddings_umap'][:, 1],
            z=data['timestamps'],
            mode='lines+markers',
            name=f'Topic {topic_id}: {topic_words}',
            text=[f"Topic: {topic_id}<br>Timestamp: {t}<br>Words: {w}" for t, w in zip(data['timestamps'], data['words'])],
            hoverinfo='text',
            visible=visible,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='Timestamp'
        ),
        width=1000,
        height=1000,
    )

    return fig

def plot_temporal_stability_metrics(temptopic, metric, darkmode=True, topics_to_show=None):
    if darkmode:
        fig = go.Figure(layout=go.Layout(template="plotly_dark"))
    else:
        fig = go.Figure()

    if topics_to_show is None or len(topics_to_show) == 0:
        topics_to_show = temptopic.final_df['Topic'].unique()

    if metric == 'topic_stability':
        df = temptopic.topic_stability_scores_df
        score_column = 'Topic Stability Score'
        title = 'Temporal Topic Stability'
    elif metric == 'representation_stability':
        df = temptopic.representation_stability_scores_df
        score_column = 'Representation Stability Score'
        title = 'Temporal Representation Stability'
    else:
        raise ValueError("Invalid metric. Choose 'topic_stability' or 'representation_stability'.")

    for topic_id in temptopic.final_df['Topic'].unique():
        topic_data = df[df['Topic ID'] == topic_id].sort_values(by='Start Timestamp')
        
        topic_words = temptopic.final_df[temptopic.final_df['Topic'] == topic_id]['Words'].iloc[0]
        topic_words = "_".join(topic_words.split(', ')[:3])

        x = topic_data['Start Timestamp']
        y = topic_data[score_column]

        hover_text = []
        for _, row in topic_data.iterrows():
            if metric == 'topic_stability':
                hover_text.append(f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}")
            else:
                hover_text.append(f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}<br>Representation: {row['Start Representation']}")

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=f'{topic_id}_{topic_words}',
            text=hover_text,
            hoverinfo='text',
            visible='legendonly' if topic_id not in topics_to_show else True,
            line=dict(shape='spline', smoothing=0.9),
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Timestamp',
        yaxis_title=f'{metric.replace("_", " ").capitalize()} Score',
        legend_title='Topic',
        hovermode='closest',
    )

    return fig

def plot_overall_topic_stability(temptopic, darkmode=True, normalize=False, topics_to_show=None):
    if temptopic.overall_stability_df is None:
        temptopic.calculate_overall_topic_stability()

    df = temptopic.overall_stability_df

    if topics_to_show is not None and len(topics_to_show) > 0:
        df = df[df['Topic ID'].isin(topics_to_show)]

    df = df.sort_values(by='Topic ID')

    metric_column = "Normalized Stability Score" if normalize else "Overall Stability Score"
    df['ScoreNormalized'] = df[metric_column]
    df['Color'] = df['ScoreNormalized'].apply(lambda x: px.colors.diverging.RdYlGn[int(x * (len(px.colors.diverging.RdYlGn) - 1))])

    fig = go.Figure(layout=go.Layout(template="plotly_dark" if darkmode else "plotly"))

    for _, row in df.iterrows():
        topic_id = row['Topic ID']
        metric_value = row[metric_column]
        words = temptopic.final_df[temptopic.final_df['Topic'] == topic_id]['Words'].iloc[0]
        num_timestamps = row['Number of Timestamps']
        
        fig.add_trace(go.Bar(
            x=[topic_id],
            y=[metric_value],
            marker_color=row['Color'],
            name=f'Topic {topic_id}',
            hovertext=f"Topic {topic_id}<br>Words: {words}<br>Score: {metric_value:.4f}<br>Timestamps: {num_timestamps}",
            hoverinfo='text',
            text=[num_timestamps],
            textposition='outside'
        ))

    fig.update_layout(
        title='Overall Topic Stability Scores',
        yaxis=dict(range=[0, 1]),
        yaxis_title='Overall Topic Stability Score',
        showlegend=False
    )
    
    return fig








def initialize_session_state():
    """Initialize session state variables."""
    if 'temptopic' not in st.session_state:
        st.session_state.temptopic = None
    if 'granularity' not in st.session_state:
        st.session_state.granularity = ""

def check_model_trained():
    """Check if a model is trained and display an error if not."""
    if "topic_model" not in st.session_state:
        st.error("Train a model to explore different temporal visualizations.", icon="🚨")
        st.stop()

def parameters_changed():
    """Check if any of the parameters have changed."""
    params_to_check = [
        'window_size', 'k', 'alpha', 'double_agg', 'doc_agg', 'global_agg',
        'evolution_tuning', 'global_tuning', 'granularity'
    ]
    return any(st.session_state.get(f'prev_{param}') != st.session_state.get(param) for param in params_to_check)

def display_sidebar():
    """Display the sidebar with TEMPTopic parameters."""
    with st.sidebar:
        st.header("TEMPTopic Parameters")
        
        register_widget("window_size")
        st.number_input("Window Size", min_value=2, value=2, step=1, key="window_size", on_change=save_widget_state)

        register_widget("k")
        st.number_input("Number of Nearest Embeddings (k)", 
                        min_value=1, 
                        value=1, 
                        step=1, 
                        key="k", 
                        on_change=save_widget_state,
                        help="The k-th nearest neighbor used for Topic Representation Stability calculation."
                        )

        register_widget("alpha")
        st.number_input("Alpha (Topic vs Representation Stability Weight)", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.8, 
                        step=0.01, 
                        key="alpha", 
                        help="Closer to 1 gives more weight given to Topic Embedding Stability, Closer to 0 gives more weight to topic representation stability.",
                        on_change=save_widget_state)

        register_widget("double_agg")
        st.checkbox("Use Double Aggregation", value=True, key="double_agg", on_change=save_widget_state, 
                    help="If unchecked, only Document Aggregation Method will be globally used.")

        register_widget("doc_agg")
        st.selectbox("Document Aggregation Method", ["mean", "max"], key="doc_agg", on_change=save_widget_state)

        register_widget("global_agg")
        st.selectbox("Global Aggregation Method", ["max", "mean"], key="global_agg", on_change=save_widget_state)

        register_widget("evolution_tuning")
        st.checkbox("Use Evolution Tuning", value=True, key="evolution_tuning", on_change=save_widget_state)

        register_widget("global_tuning")
        st.checkbox("Use Global Tuning", value=False, key="global_tuning", on_change=save_widget_state)

def get_available_granularities(min_date, max_date):
    """Determine available time granularities based on the date range."""
    time_diff = max_date - min_date
    available_granularities = ["Day"]
    if time_diff >= pd.Timedelta(weeks=1):
        available_granularities.append("Week")
    if time_diff >= pd.Timedelta(days=30):
        available_granularities.append("Month")
    if time_diff >= pd.Timedelta(days=365):
        available_granularities.append("Year")
    return available_granularities

def select_time_granularity(available_granularities):
    """Allow user to select time granularity."""
    register_widget("granularity")
    time_granularity = st.selectbox("Select time granularity", [""] + available_granularities, key="granularity", on_change=save_widget_state)
    if time_granularity == "":
        st.info("Please select a time granularity to view the temporal visualizations.")
        st.stop()
    return time_granularity

def process_data_and_fit_temptopic(time_granularity):
    """Process data and fit TempTopic if parameters changed."""
    df = st.session_state['timefiltered_df'].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if time_granularity == "Day":
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    elif time_granularity == "Week":
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%W')
    elif time_granularity == "Month":
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m')
    elif time_granularity == 'Year':
        df['timestamp'] = df['timestamp'].dt.strftime('%Y')

    aggregated_df = df.groupby('timestamp').agg({TEXT_COLUMN: list, 'index': list}).reset_index()

    indices = st.session_state["timefiltered_df"]["index"]
    docs = [st.session_state["split_df"][TEXT_COLUMN][i] for i in indices]

    index_to_timestamp = {idx: timestamp for timestamp, idx_sublist in zip(aggregated_df['timestamp'], aggregated_df['index']) for idx in idx_sublist}
    timestamps_repeated = [index_to_timestamp[idx] for idx in indices]
    
    # Initialize and fit TempTopic
    with st.spinner("Fitting TempTopic..."):
        temptopic = TempTopic(
            st.session_state['topic_model'],
            docs,
            st.session_state['embeddings'],
            st.session_state['token_embeddings'],
            st.session_state['token_strings'],
            timestamps_repeated,
            evolution_tuning=st.session_state.evolution_tuning,
            global_tuning=st.session_state.global_tuning
        )
        temptopic.fit(
            window_size=st.session_state.window_size,
            k=st.session_state.k,
            double_agg=st.session_state.double_agg,
            doc_agg=st.session_state.doc_agg,
            global_agg=st.session_state.global_agg
        )

    # Store the fitted TempTopic object and current parameter values
    st.session_state.temptopic = temptopic
    st.session_state.aggregated_df = aggregated_df
    for param in ['window_size', 'k', 'alpha', 'double_agg', 'doc_agg', 'global_agg', 'evolution_tuning', 'global_tuning', 'granularity']:
        st.session_state[f'prev_{param}'] = st.session_state.get(param)

def display_topic_evolution_dataframe():
    """Display the Topic Evolution Dataframe."""
    with st.expander("Topic Evolution Dataframe"):
        columns_to_display = ["Topic", "Words", "Frequency", "Timestamp"]
        columns_present = [col for col in columns_to_display if col in st.session_state.temptopic.final_df.columns]
        st.dataframe(st.session_state.temptopic.final_df[columns_present].sort_values(by=['Topic', 'Timestamp'], ascending=[True, True]), use_container_width=True)

def display_topic_info_dataframe():
    """Display the Topic Info Dataframe."""
    with st.expander("Topic Info Dataframe"):
        st.dataframe(st.session_state.temptopic.topic_model.get_topic_info(), use_container_width=True)

def display_documents_per_date_dataframe():
    """Display the Documents per Date Dataframe."""
    with st.expander("Documents per Date Dataframe"):
        st.dataframe(st.session_state.aggregated_df, use_container_width=True)

def display_temptopic_visualizations():
    """Display TempTopic Visualizations."""
    with st.expander("TempTopic Visualizations"):
        topics_to_show = st.multiselect("Topics to Show", options=list(st.session_state.temptopic.final_df["Topic"].unique()), default=None)

        display_topic_evolution(topics_to_show)
        display_overall_topic_stability(topics_to_show)
        display_temporal_stability_metrics(topics_to_show)

def display_topic_evolution(topics_to_show):
    """Display Topic Evolution in Time and Semantic Space."""
    st.header("Topic Evolution in Time and Semantic Space")
    n_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=100, value=15, step=1)
    min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
    metric = st.selectbox("UMAP Metric", ["cosine", "euclidean", "manhattan"])
    color_palette = st.selectbox("Color Palette", ["Plotly", "D3", "Alphabet"])
    
    fig_topic_evolution = plot_topic_evolution(
        st.session_state.temptopic,
        granularity=st.session_state.granularity, 
        topics_to_show=topics_to_show,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        color_palette=color_palette
    )
    st.plotly_chart(fig_topic_evolution, use_container_width=True)
    
    st.divider()

def display_overall_topic_stability(topics_to_show):
    """Display Overall Topic Stability."""
    st.header("Overall Topic Stability")
    normalize_overall_stability = st.checkbox("Normalize", value=False)
    overall_stability_df = st.session_state.temptopic.calculate_overall_topic_stability(window_size=st.session_state.window_size, k=st.session_state.k, alpha=st.session_state.alpha)
    fig_overall_stability = plot_overall_topic_stability(
        st.session_state.temptopic,
        topics_to_show=topics_to_show, 
        normalize=normalize_overall_stability,
        darkmode=True
    )
    st.plotly_chart(fig_overall_stability, use_container_width=True)

    st.divider()

def display_temporal_stability_metrics(topics_to_show):
    """Display Temporal Stability Metrics."""
    st.header("Temporal Stability Metrics")
    col1, col2 = st.columns(2)

    with col1:
        fig_topic_stability = plot_temporal_stability_metrics(
            st.session_state.temptopic,
            metric="topic_stability", 
            topics_to_show=topics_to_show
        )
        st.plotly_chart(fig_topic_stability, use_container_width=True)

    with col2:
        fig_representation_stability = plot_temporal_stability_metrics(
            st.session_state.temptopic,
            metric="representation_stability", 
            topics_to_show=topics_to_show
        )
        st.plotly_chart(fig_representation_stability, use_container_width=True)

def display_topics_popularity():
    """Display the popularity of topics over time."""
    with st.spinner("Computing topics over time..."):
        with st.expander("Popularity of topics over time"):
            if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"]:
                st.write("## Popularity of topics over time")

                # Parameters
                st.text_input(
                    "Topics list (format 1,12,52 or 1:20)",
                    key="dynamic_topics_list",
                    value="0:10",
                )

                st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

                # Compute topics over time
                st.session_state["topics_over_time"] = compute_topics_over_time(
                    st.session_state["parameters"],
                    st.session_state["topic_model"],
                    st.session_state["timefiltered_df"],
                    nr_bins=st.session_state["nr_bins"],
                )

                # Visualize
                st.plotly_chart(plot_topics_over_time(
                    st.session_state["topics_over_time"],
                    st.session_state["dynamic_topics_list"],
                    st.session_state["topic_model"],
                ), use_container_width=True)







def main():
    """Main function to run the Streamlit app."""
    # Restore widget state
    restore_widget_state()

    # Initialize session state
    initialize_session_state()

    # Check if model is trained
    check_model_trained()

    # Display sidebar
    display_sidebar()

    # Get available granularities
    min_date = st.session_state['timefiltered_df']['timestamp'].min()
    max_date = st.session_state['timefiltered_df']['timestamp'].max()
    available_granularities = get_available_granularities(min_date, max_date)

    # Select time granularity
    time_granularity = select_time_granularity(available_granularities)

    # Process data and fit TempTopic if parameters changed
    if parameters_changed() or 'temptopic' not in st.session_state:
        process_data_and_fit_temptopic(time_granularity)
    else:
        st.session_state.temptopic = st.session_state.temptopic 
        st.session_state.aggregated_df = st.session_state.aggregated_df

    # Display visualizations
    if st.session_state.temptopic is not None:
        display_topic_evolution_dataframe()
        display_topic_info_dataframe()
        display_documents_per_date_dataframe()
        display_temptopic_visualizations()
        display_topics_popularity()
    else:
        st.info("Please select a time granularity to view the temporal visualizations.")

if __name__ == "__main__":
    main()
    
# FIXME: Popularity of topics over time visualization is based on the number of paragraphs instead of original articles, since it's the default BERTopic method
