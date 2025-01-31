import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# Column definitions
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "source_doc"
CONTEXT_COLUMN = "context"
COMPLEXITY_COLUMN = "complexity"
RAG_RELEVANT_EXTRACTS_COLUMN = "relevant_extracts"
RAG_QUERY_TIME_COLUMN = "rag_query_time_seconds"  # Added timing column
RAG_RETRIEVER_TIME_COLUMN = "rag_retriever_time_seconds"  # Added timing column

def get_available_metrics(df: pd.DataFrame) -> list:
    """Get list of available metrics from DataFrame columns"""
    metrics = []
    for metric in ['faithfulness', 'correctness', 'retrievability']:
        if f'{metric}_score' in df.columns:
            metrics.append(metric)
    return metrics

def load_evaluation_files(eval_dir: str = "evaluation_results"):
    """Load and combine all evaluation Excel files from the directory"""
    eval_path = Path(eval_dir)
    excel_files = list(eval_path.glob("*.xlsx"))
    
    all_dfs = {}
    combined_df = None
    
    for file in excel_files:
        if "combined" not in file.name:
            model_name = file.stem.split('_')[-1]
            df = pd.read_excel(file)
            all_dfs[model_name] = df
            
            if combined_df is None:
                combined_df = df.copy()
                combined_df.columns = [f"{col}_{model_name}" if col != "question" else col 
                                    for col in combined_df.columns]
            else:
                rename_cols = {col: f"{col}_{model_name}" for col in df.columns if col != "question"}
                df_renamed = df.rename(columns=rename_cols)
                combined_df = combined_df.merge(df_renamed, on="question", how="outer")
    
    return all_dfs, combined_df

def calculate_good_score_percentage(scores):
    """Calculate percentage of good scores (4-5) in the series"""
    if scores is None or len(scores) == 0:
        return 0
    good_scores = scores[scores.isin([4, 5])].count()
    total_scores = scores.count()
    return (good_scores / total_scores * 100) if total_scores > 0 else 0

def create_timing_boxplot(all_dfs: dict):
    """Create box plot comparing query and retriever times"""
    fig = go.Figure()
    
    for model_name, df in all_dfs.items():
        if RAG_QUERY_TIME_COLUMN in df.columns:
            fig.add_trace(go.Box(
                y=df[RAG_QUERY_TIME_COLUMN],
                name=f"{model_name} (Query)",
                boxpoints='all'
            ))
        if RAG_RETRIEVER_TIME_COLUMN in df.columns:
            fig.add_trace(go.Box(
                y=df[RAG_RETRIEVER_TIME_COLUMN],
                name=f"{model_name} (Retriever)",
                boxpoints='all'
            ))
    
    fig.update_layout(
        title="RAG Timing Analysis",
        yaxis_title="Time (seconds)",
        height=500
    )
    
    return fig

def create_metric_comparison_plot(all_dfs: dict, metric: str):
    """Create a violin plot comparing a specific metric across judges"""
    fig = go.Figure()
    
    for model_name, df in all_dfs.items():
        metric_col = f"{metric}_score"
        if metric_col in df.columns:
            fig.add_trace(go.Violin(
                y=df[metric_col],
                name=model_name,
                box_visible=True,
                meanline_visible=True,
                points='all'
            ))
    
    if len(fig.data) == 0:
        return None
    
    fig.update_layout(
        title=f"{metric.title()} Score Distribution Across Judges",
        yaxis_title=f"{metric.title()} Score",
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 5.5]
        ),
        height=500
    )
    
    return fig

def create_score_distribution_plot(df: pd.DataFrame, model_name: str):
    """Create a bar plot of score distributions for a specific judge"""
    available_metrics = get_available_metrics(df)
    if not available_metrics:
        return None
        
    fig = go.Figure()
    
    for metric in available_metrics:
        metric_col = f'{metric}_score'
        score_counts = df[metric_col].value_counts().sort_index()
        
        colors = ['rgb(239, 85, 59)' if score <= 3 else 'rgb(99, 110, 250)' 
                 for score in score_counts.index]
        
        fig.add_trace(go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            name=metric.title(),
            opacity=0.7,
            marker_color=colors
        ))
    
    fig.update_layout(
        title=f"Score Distribution for Judge: {model_name}<br><sup>(Scores 1-3 in red, 4-5 in blue)</sup>",
        xaxis_title="Score",
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis_title="Number of Evaluations",
        barmode='group',
        height=400
    )
    
    return fig

def create_radar_plot(all_dfs: dict):
    """Create a radar plot comparing good score percentages across judges"""
    fig = go.Figure()
    
    all_metrics = set()
    for df in all_dfs.values():
        all_metrics.update(get_available_metrics(df))
    
    if not all_metrics:
        return None
        
    metrics = sorted(list(all_metrics))
    
    for model_name, df in all_dfs.items():
        good_score_percentages = []
        for metric in metrics:
            metric_col = f'{metric}_score'
            if metric_col in df.columns:
                percentage = calculate_good_score_percentage(df[metric_col])
            else:
                percentage = 0
            good_score_percentages.append(percentage)
        
        fig.add_trace(go.Scatterpolar(
            r=good_score_percentages,
            theta=[m.title() for m in metrics],
            fill='toself',
            name=f"Judge: {model_name}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tick0=0,
                dtick=20,
                ticksuffix="%"
            )),
        showlegend=True,
        title="Performance Across Judges",
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
    
    st.title("RAG Evaluation Analysis Dashboard")
    
    eval_dir = st.text_input("Evaluation Results Directory", "evaluation_politique_voyageur")
    
    try:
        all_dfs, combined_df = load_evaluation_files(eval_dir)
        
        if not all_dfs:
            st.error("No evaluation files found in the specified directory.")
            return
        
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Overview", "Judge-Specific Analysis", "Comparative Analysis"]
        )
        
        if analysis_type == "Overview":
            st.header("Overall Evaluation Results")
            
            # Add timing overview
            timing_plot = create_timing_boxplot(all_dfs)
            st.plotly_chart(timing_plot, use_container_width=True)
            
            # Radar plot for good score percentages
            radar_plot = create_radar_plot(all_dfs)
            if radar_plot:
                st.plotly_chart(radar_plot, use_container_width=True)
            
            # Metric comparisons
            available_metrics = set()
            for df in all_dfs.values():
                available_metrics.update(get_available_metrics(df))
            
            for metric in sorted(available_metrics):
                plot = create_metric_comparison_plot(all_dfs, metric)
                if plot:
                    st.plotly_chart(plot, use_container_width=True)
        
        elif analysis_type == "Judge-Specific Analysis":
            st.header("Judge-Specific Analysis")
            
            selected_model = st.selectbox("Select Judge", list(all_dfs.keys()))
            
            # Score distribution
            dist_plot = create_score_distribution_plot(all_dfs[selected_model], selected_model)
            if dist_plot:
                st.plotly_chart(dist_plot, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Evaluation Results")
            df = all_dfs[selected_model]
            
            # Get all relevant columns including timing
            display_columns = ['question']
            
            # Add timing columns if available
            if RAG_QUERY_TIME_COLUMN in df.columns:
                display_columns.append(RAG_QUERY_TIME_COLUMN)
            if RAG_RETRIEVER_TIME_COLUMN in df.columns:
                display_columns.append(RAG_RETRIEVER_TIME_COLUMN)
            
            available_metrics = get_available_metrics(df)
            for metric in available_metrics:
                score_col = f'{metric}_score'
                reasoning_col = metric
                if score_col in df.columns:
                    display_columns.append(score_col)
                if reasoning_col in df.columns:
                    display_columns.append(reasoning_col)
            
            if len(display_columns) > 1:
                metrics_df = df[display_columns]
                st.dataframe(metrics_df, height=400)
            else:
                st.warning("No evaluation metrics found in the data")
        
        elif analysis_type == "Comparative Analysis":
            st.header("Comparative Analysis")
            
            if combined_df is not None:
                # Calculate and display timing statistics
                st.subheader("Timing Statistics")
                timing_stats = []
                for model_name, df in all_dfs.items():
                    stats = {'Judge': model_name}
                    if RAG_QUERY_TIME_COLUMN in df.columns:
                        stats.update({
                            'Avg Query Time (s)': f"{df[RAG_QUERY_TIME_COLUMN].mean():.3f}",
                            'Max Query Time (s)': f"{df[RAG_QUERY_TIME_COLUMN].max():.3f}"
                        })
                    if RAG_RETRIEVER_TIME_COLUMN in df.columns:
                        stats.update({
                            'Avg Retriever Time (s)': f"{df[RAG_RETRIEVER_TIME_COLUMN].mean():.3f}",
                            'Max Retriever Time (s)': f"{df[RAG_RETRIEVER_TIME_COLUMN].max():.3f}"
                        })
                    timing_stats.append(stats)
                
                if timing_stats:
                    st.dataframe(pd.DataFrame(timing_stats))
                
                # Performance summary
                st.subheader("Performance Summary")
                stats_data = []
                for model_name in all_dfs.keys():
                    df = all_dfs[model_name]
                    available_metrics = get_available_metrics(df)
                    
                    model_stats = {'Judge': model_name}
                    for metric in available_metrics:
                        metric_col = f'{metric}_score'
                        if metric_col in df.columns:
                            model_stats[f'Performance {metric.title()} %'] = f"{calculate_good_score_percentage(df[metric_col]):.1f}%"
                    
                    model_stats['Number of Evaluations'] = len(df)
                    stats_data.append(model_stats)
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, height=200)
                
                st.subheader("Detailed Combined Results")
                st.dataframe(combined_df, height=400)
    
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")

if __name__ == "__main__":
    main()