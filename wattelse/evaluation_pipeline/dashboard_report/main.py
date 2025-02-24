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
    create_metrics_summary,  # Import the new function
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
                'dir': 'AZURE-5',
                'name': 'Top-5-extracts'
            },
            {
                'dir': 'AZURE-10',
                'name': 'Top-10-extracts'
            },
            {
                'dir': 'AZURE-15',
                'name': 'Top-15-extracts'
            },
            {
                'dir': 'AZURE-20',
                'name': 'Top-20-extracts'
            },
        ]

    # Sidebar navigation
    page = st.sidebar.radio(
        "Select a Page",
        ["Experiment Setup", "Performance Overview", "Timing Analysis", "Raw Data"]
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
        
        # Create tabs for Summary and Individual Judges
        tab_summary, *judge_tabs = st.tabs(["Summary"] + list(sorted(all_judges)))
        
        with tab_summary:
            st.subheader("Evaluation Summary")
            st.caption("Average good score percentages (scores of 4-5) across all LLM judges")
            
            # Generate summary metrics
            summary_dfs, summary_figs, overall_df, overall_fig = create_metrics_summary(experiments_data)
            
            # Display overall summary
            st.plotly_chart(overall_fig, use_container_width=True)
            
            # Display overall summary table with highest scores in bold
            st.subheader("Summary Table")
            formatted_df = overall_df.copy()
            
            # Format each column with highest value in bold
            for col in formatted_df.columns:
                if col != 'Experiment':
                    # Find max value in this column
                    max_val = formatted_df[col].max()
                    # Create display column with bold formatting for maximum values
                    formatted_df[f"{col}_display"] = formatted_df[col].apply(
                        lambda x: f"**{x:.1f}%**" if x == max_val else f"{x:.1f}%"
                    )
            
            # Create a new DataFrame with just the display columns
            display_df = pd.DataFrame()
            display_df['Experiment'] = formatted_df['Experiment']
            
            # Add formatted display columns in the right order
            for col in overall_df.columns:
                if col != 'Experiment':
                    display_df[col] = formatted_df[f"{col}_display"]
            
            # Use st.markdown to render the bold formatting
            st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
            
            # Display individual metric summaries in expandable sections
            st.subheader("Individual Metric Summaries")
            for metric in sorted(summary_figs.keys()):
                with st.expander(f"{metric.title()} Metric Details", expanded=False):
                    st.plotly_chart(summary_figs[metric], use_container_width=True)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        # Create display DataFrame with bold formatting
                        display_df = pd.DataFrame()
                        display_df['Experiment'] = summary_dfs[metric]['Experiment']
                        display_df['Average Good Score %'] = summary_dfs[metric]['Display Score']
                        display_df['Judges Count'] = summary_dfs[metric]['Judges Count']
                        
                        # Use st.markdown to render the bold formatting
                        st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
                    with col2:
                        if metric in METRIC_DESCRIPTIONS:
                            st.info(METRIC_DESCRIPTIONS[metric])
        

        
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
                    
                    # Add a brief explanation of the metrics expandable sections
                    st.info("Click on each metric below to see detailed comparison across experiments")
                    
                    for metric in sorted(metrics):
                        # Use markdown with HTML for larger, more prominent title in the expander
                        with st.expander(f"### {metric.title()} Metric", expanded=False):
                            if metric in METRIC_DESCRIPTIONS:
                                st.info(METRIC_DESCRIPTIONS[metric])
                            
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
                            
                            # Format with bold for highest score
                            display_df = pd.DataFrame()
                            display_df['Experiment'] = metric_df['Experiment']
                            
                            # Get the highest score percentage
                            max_score = max([float(score.rstrip('%')) for score in metric_df['Good Score %']])
                            
                            # Create formatted column with bold for maximum values
                            display_df['Good Score %'] = metric_df['Good Score %'].apply(
                                lambda x: f"**{x}**" if float(x.rstrip('%')) == max_score else x
                            )
                            
                            # Use H3 tag for consistent heading style - same size as the main metric title
                            st.markdown(f"##### {metric.title()} Metric Analysis")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Use st.markdown to render the bold formatting
                                st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
                            
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
                            # Use H3 tag for consistent heading size
                            st.markdown(f"##### {metric.title()} Score Distribution")
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
        tab1, tab2 = st.tabs(["📊 Total Time", "🔄 Retriever Time"])
        
        with tab1:
            fig = create_timing_plot(experiments_data, RAG_QUERY_TIME_COLUMN, "Total Time Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_timing_plot(experiments_data, RAG_RETRIEVER_TIME_COLUMN, "Retriever Time Distribution")
            st.plotly_chart(fig, use_container_width=True)



    elif page == "Raw Data":
        st.header("Raw Data")
        
        tab1, tab2 = st.tabs(["📊 Evaluation Data", "⏱️ Timing Data"])
        
        with tab1:
            # Get all judges from all experiments
            all_judges = set()
            for exp in experiments_data:
                all_judges.update(exp['dfs'].keys())
            
            # Create experiment selection
            selected_exp = st.selectbox(
                "Select Experiment",
                [exp['name'] for exp in experiments_data]
            )
            exp_data = next(exp for exp in experiments_data if exp['name'] == selected_exp)
            
            # Create tabs for each judge
            if all_judges:
                judge_tabs = st.tabs(sorted(all_judges))
                
                for tab, judge_name in zip(judge_tabs, sorted(all_judges)):
                    with tab:
                        if judge_name in exp_data['dfs']:
                            st.subheader(f"{judge_name} Evaluation Data")
                            st.dataframe(exp_data['dfs'][judge_name], use_container_width=True)
                        else:
                            st.warning(f"No data available for {judge_name} in this experiment")
            else:
                st.warning("No judge data available")
        
        with tab2:
            for exp in experiments_data:
                st.subheader(f"{exp['name']}")
                if exp['timing'] is not None:
                    st.dataframe(exp['timing'], use_container_width=True)

if __name__ == "__main__":
    main()