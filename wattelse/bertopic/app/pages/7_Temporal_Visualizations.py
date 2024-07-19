import streamlit as st
import pandas as pd
import locale
from datetime import timedelta

from wattelse.bertopic.utils import TIMESTAMP_COLUMN, TEXT_COLUMN
from app_utils import plot_topics_over_time
from state_utils import restore_widget_state, register_widget, save_widget_state
from wattelse.bertopic.app.app_utils import plot_topics_over_time, compute_topics_over_time
from wattelse.bertopic.temporal_metrics_embedding import TempTopic

# Set locale for French date names
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')

# Page configuration
st.set_page_config(page_title="Wattelse® topic", layout="wide")

# Initialize session state variables
if 'temptopic' not in st.session_state:
    st.session_state.temptopic = None

if 'granularity' not in st.session_state:
    st.session_state.granularity = ""

# Restore widget state
restore_widget_state()

# Check if a model is trained
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

# Sidebar
with st.sidebar:
    st.header("TEMPTopic Parameters")
    
    register_widget("window_size")
    window_size = st.number_input("Window Size", min_value=2, value=2, step=1, key="window_size", on_change=save_widget_state)

    register_widget("k")
    k = st.number_input("Number of Nearest Embeddings (k)", min_value=1, value=1, step=1, key="k", on_change=save_widget_state)

    register_widget("alpha")
    alpha = st.number_input("Alpha (Topic vs Representation Stability Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="alpha", on_change=save_widget_state)

    register_widget("double_agg")
    double_agg = st.checkbox("Use Double Aggregation", value=True, key="double_agg", on_change=save_widget_state)

    register_widget("doc_agg")
    doc_agg = st.selectbox("Document Aggregation Method", ["mean", "max"], key="doc_agg", on_change=save_widget_state)

    register_widget("global_agg")
    global_agg = st.selectbox("Global Aggregation Method", ["max", "mean"], key="global_agg", on_change=save_widget_state)

    register_widget("evolution_tuning")
    evolution_tuning = st.checkbox("Use Evolution Tuning", value=True, key="evolution_tuning", on_change=save_widget_state)

    register_widget("global_tuning")
    global_tuning = st.checkbox("Use Global Tuning", value=False, key="global_tuning", on_change=save_widget_state)

# Determine available time granularities
min_date = st.session_state['timefiltered_df']['timestamp'].min()
max_date = st.session_state['timefiltered_df']['timestamp'].max()
time_diff = max_date - min_date

available_granularities = ["Day"]
if time_diff >= pd.Timedelta(weeks=1):
    available_granularities.append("Week")
if time_diff >= pd.Timedelta(days=30):
    available_granularities.append("Month")
if time_diff >= pd.Timedelta(days=365):
    available_granularities.append("Year")

# Time granularity selection
register_widget("granularity")
time_granularity = st.selectbox("Select time granularity", [""] + available_granularities, key="granularity", on_change=save_widget_state)

if time_granularity == "":
    st.info("Please select a time granularity to view the temporal visualizations.")
    st.stop()

# Process data and fit TempTopic if parameters changed
if parameters_changed() or 'temptopic' not in st.session_state:
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

else:
    temptopic = st.session_state.temptopic 
    aggregated_df = st.session_state.aggregated_df

# Visualizations
if st.session_state.temptopic is not None:
    temptopic = st.session_state.temptopic 

    # Topic Evolution Dataframe
    with st.expander("Topic Evolution Dataframe"):
        columns_to_display = ["Topic", "Words", "Frequency", "Timestamp"]
        columns_present = [col for col in columns_to_display if col in temptopic.final_df.columns]
        st.dataframe(temptopic.final_df[columns_present].sort_values(by=['Topic', 'Timestamp'], ascending=[True, True]), use_container_width=True)

    # Topic Info Dataframe
    with st.expander("Topic Info Dataframe"):
        st.dataframe(temptopic.topic_model.get_topic_info(), use_container_width=True)

    # Documents per Date Dataframe
    with st.expander("Documents per Date Dataframe"):
        st.dataframe(aggregated_df, use_container_width=True)

    # TempTopic Visualizations
    with st.expander("TempTopic Visualizations"):
        topics_to_show = st.multiselect("Topics to Show", options=list(temptopic.final_df["Topic"].unique()), default=None)

        # Topic Evolution in Time and Semantic Space
        st.header("Topic Evolution in Time and Semantic Space")
        n_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=100, value=15, step=1)
        min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
        metric = st.selectbox("UMAP Metric", ["cosine", "euclidean", "manhattan"])
        color_palette = st.selectbox("Color Palette", ["Plotly", "D3", "Alphabet"])
        
        fig_topic_evolution = temptopic.plot_topic_evolution(
            granularity=time_granularity, 
            topics_to_show=topics_to_show,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            color_palette=color_palette
        )
        st.plotly_chart(fig_topic_evolution, use_container_width=True)
        
        st.divider()

        # Overall Topic Stability
        st.header("Overall Topic Stability")
        normalize_overall_stability = st.checkbox("Normalize", value=False)
        overall_stability_df = temptopic.calculate_overall_topic_stability(window_size=window_size, k=k, alpha=alpha)
        fig_overall_stability = temptopic.plot_overall_topic_stability(
            topics_to_show=topics_to_show, 
            normalize=normalize_overall_stability,
            darkmode=True
        )
        st.plotly_chart(fig_overall_stability, use_container_width=True)

        st.divider()
        
        # Temporal Stability Metrics
        st.header("Temporal Stability Metrics")
        col1, col2 = st.columns(2)

        with col1:
            fig_topic_stability = temptopic.plot_temporal_stability_metrics(
                metric="topic_stability", 
                topics_to_show=topics_to_show
            )
            st.plotly_chart(fig_topic_stability, use_container_width=True)

        with col2:
            fig_representation_stability = temptopic.plot_temporal_stability_metrics(
                metric="representation_stability", 
                topics_to_show=topics_to_show
            )
            st.plotly_chart(fig_representation_stability, use_container_width=True)

    # Popularity of topics over time
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
else:
    st.info("Please select a time granularity to view the temporal visualizations.")
    
    
# FIXME: Popularity of topics over time visualization is based on the number of paragraphs instead of original articles, since it's the default BERTopic method
