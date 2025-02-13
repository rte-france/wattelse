"""Main Streamlit application."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import (
    load_evaluation_files, 
    get_available_metrics,
    calculate_good_score_percentage,
    create_timing_plot, 
    create_radar_plot,
    create_score_distribution_plot,
    RAG_QUERY_TIME_COLUMN, 
    RAG_RETRIEVER_TIME_COLUMN, 
    METRIC_DESCRIPTIONS
)

def setup_page():
    """Configure page settings and title."""
    st.set_page_config(page_title="RAG Experiment Comparison", layout="wide")
    st.title("RAG Evaluation Pipeline Dashboard")

def handle_experiment_setup():
    """Handle experiment setup page."""
    st.header("Experiment Configuration")
    
    st.info("""
    Configure the experiments you want to compare:
    1. Enter the directory path containing your evaluation Excel files
    2. Give each experiment a meaningful name
    3. Add more experiments using the '+' button below
    """)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add Experiment"):
            st.session_state.experiments.append({
                'dir': '',
                'name': f'Experiment {len(st.session_state.experiments) + 1}'
            })
    
    for i, exp in enumerate(st.session_state.experiments):
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 0.5])
            with col1:
                exp['dir'] = st.text_input("📁 Directory", value=exp['dir'], key=f"dir_{i}")
            with col2:
                exp['name'] = st.text_input("📝 Name", value=exp['name'], key=f"name_{i}")
            with col3:
                if st.button("🗑️", key=f"remove_{i}", help="Remove this experiment"):
                    st.session_state.experiments.pop(i)
                    st.rerun()
            st.divider()

def display_metric_descriptions():
    """Display metric descriptions in an expandable section."""
    with st.expander("ℹ️ Metric Descriptions", expanded=False):
        for metric, description in METRIC_DESCRIPTIONS.items():
            st.markdown(f"**{metric.title()}**: {description}")

def main():
    setup_page()

    # Initialize session state
    if 'experiments' not in st.session_state:
        st.session_state.experiments = [
            {
                'dir': 'eval_GPT4o-mini',
                'name': 'GPT4o-mini Experiment'
            },
            {
                'dir': 'eval_llama3-8B',
                'name': 'Llama3-8B Experiment'
            }
        ]

    # Sidebar navigation
    page = st.sidebar.radio(
        "Select a Page",
        ["Experiment Setup", "Performance Overview", "Timing Analysis", "Score Analysis", "Raw Data"]
    )

    if page == "Experiment Setup":
        handle_experiment_setup()
        return

    # Load experiments data
    experiments_data = []
    has_invalid_paths = False
    
    for exp in st.session_state.experiments:
        data = load_evaluation_files(exp['dir'])
        if data is not None:
            experiments_data.append({
                'name': exp['name'],
                'dfs': data[0],
                'combined': data[1],
                'timing': data[2]
            })
        else:
            has_invalid_paths = True

    # Handle various error states
    if not st.session_state.experiments:
        st.error("No experiments configured. Please add experiments in the Setup page.")
        return
    elif has_invalid_paths:
        st.error("Some experiment paths are invalid or empty. Please check the configuration.")
        return
    elif not experiments_data:
        st.error("No valid evaluation files found in the specified directories.")
        return

    if page == "Performance Overview":
        st.header("Performance Overview")
        
        # Display metric descriptions at the top of the performance page
        display_metric_descriptions()
        
        # Get all judges
        all_judges = set()
        for exp in experiments_data:
            all_judges.update(exp['dfs'].keys())
        
        # Create tabs for Overview and Individual Judges
        tab_overview, *judge_tabs = st.tabs(["Overall View"] + list(sorted(all_judges)))
        
        with tab_overview:
            # Overall radar plot
            st.subheader("Overall Radar Plot Analysis")
            st.caption("Hover over the radar plot to see detailed scores. Each axis represents a different evaluation metric.")
            fig = create_radar_plot(experiments_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Create individual judge tabs
        for judge_tab, judge_name in zip(judge_tabs, sorted(all_judges)):
            with judge_tab:
                st.subheader(f"Analysis by {judge_name}")
                
                # Filter data for this judge
                judge_data = []
                for exp in experiments_data:
                    if judge_name in exp['dfs']:
                        judge_data.append({
                            'name': exp['name'],
                            'dfs': {judge_name: exp['dfs'][judge_name]},
                            'combined': exp['combined'],
                            'timing': exp['timing']
                        })
                
                if judge_data:
                    # Create radar plot for this judge
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = create_radar_plot(judge_data)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add experiment comparison
                    st.subheader("Experiment Comparison")
                    metrics = [col.replace('_score', '') 
                             for col in judge_data[0]['dfs'][judge_name].columns 
                             if col.endswith('_score')]
                    
                    for metric in sorted(metrics):
                        st.write(f"**{metric.title()} Metric**")
                        if metric in METRIC_DESCRIPTIONS:
                            st.caption(METRIC_DESCRIPTIONS[metric])
                        
                        # Prepare data for both table and plot
                        metric_data = []
                        plot_data = []
                        for exp_data in judge_data:
                            df = exp_data['dfs'][judge_name]
                            score_col = f'{metric}_score'
                            if score_col in df.columns:
                                good_score_pct = calculate_good_score_percentage(df[score_col])
                                metric_data.append({
                                    'Experiment': exp_data['name'],
                                    'Good Score %': f"{good_score_pct:.1f}%",
                                })
                                # Add all scores for the plot
                                scores = df[score_col].value_counts().sort_index()
                                for score, count in scores.items():
                                    plot_data.append({
                                        'Experiment': exp_data['name'],
                                        'Score': score,
                                        'Count': count,
                                        'Percentage': (count / len(df[score_col])) * 100
                                    })
                        
                        # Create and display table with good score plot side by side
                        metric_df = pd.DataFrame(metric_data)
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.dataframe(metric_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            # Create good score percentage plot
                            good_score_fig = go.Figure()
                            
                            # Extract experiment names and good judgment percentages
                            experiments = [row['Experiment'] for row in metric_data]
                            good_judgments = [float(row['Good Score %'].rstrip('%')) for row in metric_data]
                            
                            good_score_fig.add_trace(go.Scatter(
                                x=experiments,
                                y=good_judgments,
                                mode='lines+markers',
                                name='Good Judgments %',
                                line=dict(width=3),
                                marker=dict(size=10),
                                hovertemplate="Experiment: %{x}<br>Good Judgments: %{y:.1f}%<extra></extra>"
                            ))
                            
                            good_score_fig.update_layout(
                                yaxis_title="Good Judgments Percentage",
                                yaxis_ticksuffix="%",
                                yaxis_range=[0, 100],
                                showlegend=False,
                                height=200,
                                margin=dict(t=0, b=0, l=0, r=0)
                            )
                            st.plotly_chart(good_score_fig, use_container_width=True)
                        
                        # Create and display score distribution plot
                        plot_df = pd.DataFrame(plot_data)
                        if not plot_df.empty:
                            fig = go.Figure()
                            
                            # Add traces for each experiment
                            for exp_name in plot_df['Experiment'].unique():
                                exp_data = plot_df[plot_df['Experiment'] == exp_name]
                                fig.add_trace(go.Scatter(
                                    x=exp_data['Score'],
                                    y=exp_data['Percentage'],
                                    mode='lines+markers',
                                    name=exp_name,
                                    hovertemplate="Score: %{x}<br>Percentage: %{y:.1f}%<extra></extra>"
                                ))
                            
                            fig.update_layout(
                                xaxis_title="Judgment (1-5)",
                                yaxis_title="Percentage of Responses",
                                yaxis_ticksuffix="%",
                                xaxis=dict(
                                    tickmode='linear',
                                    tick0=1,
                                    dtick=1,
                                    range=[0.5, 5.5]
                                ),
                                hovermode='x unified',
                                showlegend=True,
                                height=400,
                                margin=dict(t=0, b=0, l=0, r=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {judge_name}")

    elif page == "Timing Analysis":
        st.header("Timing Analysis")
        tab1, tab2 = st.tabs(["📊 Query Time", "🔄 Retriever Time"])
        
        with tab1:
            fig = create_timing_plot(experiments_data, RAG_QUERY_TIME_COLUMN, "Query Time Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_timing_plot(experiments_data, RAG_RETRIEVER_TIME_COLUMN, "Retriever Time Distribution")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Score Analysis":
        st.header("LLM Jury Analysis")
        
        # Get all available metrics
        all_metrics = set()
        for exp in experiments_data:
            for df in exp['dfs'].values():
                all_metrics.update(get_available_metrics(df))

        # Metric selection
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_metric = st.selectbox("Metric", options=sorted(all_metrics))
        with col2:
            if selected_metric in METRIC_DESCRIPTIONS:
                st.info(METRIC_DESCRIPTIONS[selected_metric])

        if selected_metric:
            # Create plots
            plot_tab1, plot_tab2 = st.tabs(["Violin Plot", "Box Plot"])
            
            with plot_tab1:
                fig = create_score_distribution_plot(experiments_data, selected_metric, "violin")
                st.plotly_chart(fig, use_container_width=True)
            
            with plot_tab2:
                fig = create_score_distribution_plot(experiments_data, selected_metric, "box")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Key Findings:**
            - Violin plots show the full distribution of scores from each jury member
            - Box plots highlight the median, quartiles, and any outliers
            - Left/Right split in violin plots helps compare experiments directly
            """)

    elif page == "Raw Data":
        st.header("Raw Data")
        
        tab1, tab2 = st.tabs(["📊 Evaluation Data", "⏱️ Timing Data"])
        
        with tab1:
            selected_exp = st.selectbox(
                "Select Experiment",
                [exp['name'] for exp in experiments_data]
            )
            exp_data = next(exp for exp in experiments_data if exp['name'] == selected_exp)
            st.dataframe(exp_data['combined'], use_container_width=True)
        
        with tab2:
            for exp in experiments_data:
                st.subheader(f"{exp['name']}")
                if exp['timing'] is not None:
                    st.dataframe(exp['timing'], use_container_width=True)

if __name__ == "__main__":
    main()