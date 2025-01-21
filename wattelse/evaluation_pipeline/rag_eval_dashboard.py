# rag_eval_dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import glob

def load_evaluation_files(eval_dir: str = "evaluation_results"):
    """Load and combine all evaluation Excel files from the directory"""
    eval_path = Path(eval_dir)
    excel_files = list(eval_path.glob("*.xlsx"))
    
    all_dfs = {}
    combined_df = None
    
    for file in excel_files:
        if "combined" not in file.name:  # Skip combined results file for individual analysis
            model_name = file.stem.split('_')[-1]
            df = pd.read_excel(file)
            all_dfs[model_name] = df
            
            # Process for combined view
            if combined_df is None:
                combined_df = df.copy()
                combined_df.columns = [f"{col}_{model_name}" if col != "question" else col 
                                    for col in combined_df.columns]
            else:
                # Rename columns to include model name and merge
                rename_cols = {col: f"{col}_{model_name}" for col in df.columns if col != "question"}
                df_renamed = df.rename(columns=rename_cols)
                combined_df = combined_df.merge(df_renamed, on="question", how="outer")
    
    return all_dfs, combined_df

def create_metric_comparison_plot(all_dfs: dict, metric: str):
    """Create a box plot comparing a specific metric across models"""
    fig = go.Figure()
    
    for model_name, df in all_dfs.items():
        fig.add_trace(go.Box(
            y=df[f"{metric}_score"],
            name=model_name,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title=f"{metric.title()} Score Comparison Across Models",
        yaxis_title=f"{metric.title()} Score",
        boxmode='group',
        height=500
    )
    
    return fig

def create_score_distribution_plot(df: pd.DataFrame, model_name: str):
    """Create a histogram of score distributions for a specific model"""
    metrics = ['faithfulness_score', 'correctness_score', 'retrievability_score']
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Histogram(
            x=df[metric],
            name=metric.replace('_score', '').title(),
            nbinsx=10,
            opacity=0.7
        ))
    
    fig.update_layout(
        title=f"Score Distribution for {model_name}",
        xaxis_title="Score",
        yaxis_title="Count",
        barmode='overlay',
        height=400
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, model_name: str):
    """Create a correlation heatmap for evaluation metrics"""
    score_cols = [col for col in df.columns if 'score' in col]
    corr_matrix = df[score_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        text=corr_matrix.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=f"Metric Correlation Heatmap - {model_name}",
        height=400
    )
    
    return fig

def create_radar_plot(all_dfs: dict):
    """Create a radar plot comparing average metrics across models"""
    fig = go.Figure()
    
    metrics = ['faithfulness_score', 'correctness_score', 'retrievability_score']
    
    for model_name, df in all_dfs.items():
        avg_scores = [df[metric].mean() for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=avg_scores,
            theta=[m.replace('_score', '').title() for m in metrics],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Average Metrics Comparison",
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
    
    st.title("RAG Evaluation Analysis Dashboard")
    
    # Load data
    eval_dir = st.text_input("Evaluation Results Directory", "evaluation_results")
    
    try:
        all_dfs, combined_df = load_evaluation_files(eval_dir)
        
        if not all_dfs:
            st.error("No evaluation files found in the specified directory.")
            return
        
        # Sidebar for navigation
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Overview", "Model-Specific Analysis", "Comparative Analysis"]
        )
        
        if analysis_type == "Overview":
            st.header("Overall Evaluation Results")
            
            # Radar plot for average metrics
            st.plotly_chart(create_radar_plot(all_dfs), use_container_width=True)
            
            # Metric comparisons
            for metric in ['faithfulness', 'correctness', 'retrievability']:
                st.plotly_chart(
                    create_metric_comparison_plot(all_dfs, metric),
                    use_container_width=True
                )
        
        elif analysis_type == "Model-Specific Analysis":
            st.header("Model-Specific Analysis")
            
            selected_model = st.selectbox("Select Model", list(all_dfs.keys()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                st.plotly_chart(
                    create_score_distribution_plot(all_dfs[selected_model], selected_model),
                    use_container_width=True
                )
            
            with col2:
                # Correlation heatmap
                st.plotly_chart(
                    create_correlation_heatmap(all_dfs[selected_model], selected_model),
                    use_container_width=True
                )
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_df = all_dfs[selected_model][[
                'question',
                'faithfulness_score',
                'correctness_score',
                'retrievability_score'
            ]]
            st.dataframe(metrics_df, height=400)
        
        elif analysis_type == "Comparative Analysis":
            st.header("Comparative Analysis")
            
            # Display combined results
            if combined_df is not None:
                st.subheader("Combined Results Overview")
                st.dataframe(combined_df, height=400)
                
                # Calculate and display aggregate statistics
                st.subheader("Aggregate Statistics")
                
                stats_data = []
                for model_name in all_dfs.keys():
                    model_stats = {
                        'Model': model_name,
                        'Avg Faithfulness': all_dfs[model_name]['faithfulness_score'].mean(),
                        'Avg Correctness': all_dfs[model_name]['correctness_score'].mean(),
                        'Avg Retrievability': all_dfs[model_name]['retrievability_score'].mean(),
                        'Sample Size': len(all_dfs[model_name])
                    }
                    stats_data.append(model_stats)
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, height=200)
    
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")

if __name__ == "__main__":
    main()