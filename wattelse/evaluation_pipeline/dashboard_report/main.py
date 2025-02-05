"""Main Streamlit application."""

import streamlit as st
from utils import (
    load_evaluation_files, 
    get_available_metrics,
    create_performance_summary,
    create_timing_plot, 
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

def main():
    setup_page()

    # Initialize session state
    if 'experiments' not in st.session_state:
        st.session_state.experiments = [
            {
                'dir': 'eval_GPT4o-mini',
                'name': 'GPT4 Mini Experiment'
            },
            {
                'dir': 'eval_llama3-8B',
                'name': 'Llama3 8B Experiment'
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

    # Render appropriate page content
    if page == "Performance Overview":
        st.header("Performance Overview")
        summary_df = create_performance_summary(experiments_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=400)

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