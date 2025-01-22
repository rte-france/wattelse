import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

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
    good_scores = scores[scores.isin([4, 5])].count()
    total_scores = scores.count()
    return (good_scores / total_scores * 100) if total_scores > 0 else 0

def create_metric_comparison_plot(all_dfs: dict, metric: str):
    """Create a violin plot comparing a specific metric across judges"""
    fig = go.Figure()
    
    for model_name, df in all_dfs.items():
        fig.add_trace(go.Violin(
            y=df[f"{metric}_score"],
            name=model_name,
            box_visible=True,
            meanline_visible=True,
            points='all'
        ))
    
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
    metrics = ['faithfulness_score', 'correctness_score', 'retrievability_score']
    
    fig = go.Figure()
    for metric in metrics:
        score_counts = df[metric].value_counts().sort_index()
        
        # Color bars based on good/bad scores
        colors = ['rgb(239, 85, 59)' if score <= 3 else 'rgb(99, 110, 250)' 
                 for score in score_counts.index]
        
        fig.add_trace(go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            name=metric.replace('_score', '').title(),
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
    
    metrics = ['faithfulness_score', 'correctness_score', 'retrievability_score']
    
    for model_name, df in all_dfs.items():
        good_score_percentages = [calculate_good_score_percentage(df[metric]) for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=good_score_percentages,
            theta=[m.replace('_score', '').title() for m in metrics],
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
    
    eval_dir = st.text_input("Evaluation Results Directory", "evaluation_results")
    
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
            
            # Radar plot for good score percentages
            st.plotly_chart(create_radar_plot(all_dfs), use_container_width=True)
            
            # Metric comparisons
            for metric in ['faithfulness', 'correctness', 'retrievability']:
                st.plotly_chart(
                    create_metric_comparison_plot(all_dfs, metric),
                    use_container_width=True
                )
        
        elif analysis_type == "Judge-Specific Analysis":
            st.header("Judge-Specific Analysis")
            
            selected_model = st.selectbox("Select Judge", list(all_dfs.keys()))
            
            # Score distribution
            st.plotly_chart(
                create_score_distribution_plot(all_dfs[selected_model], selected_model),
                use_container_width=True
            )
            
            # Detailed metrics table with reasoning
            st.subheader("Detailed Evaluation Results")
            # Get all columns from the DataFrame
            df_columns = all_dfs[selected_model].columns
            
            # Start with required columns
            display_columns = ['question']
            
            # Add score and reasoning columns if they exist
            for metric in ['faithfulness', 'correctness', 'retrievability']:
                score_col = f'{metric}_score'
                reasoning_col = f'{metric}'
                
                if score_col in df_columns:
                    display_columns.append(score_col)
                if reasoning_col in df_columns:
                    display_columns.append(reasoning_col)
            
            metrics_df = all_dfs[selected_model][display_columns]
            st.dataframe(metrics_df, height=400)
        
        elif analysis_type == "Comparative Analysis":
            st.header("Comparative Analysis")
            
            if combined_df is not None:
                # Calculate and display aggregate statistics
                st.subheader("Performance Summary")
                
                stats_data = []
                for model_name in all_dfs.keys():
                    df = all_dfs[model_name]
                    model_stats = {
                        'Judge': model_name,
                        'Performance Faithfulness %': f"{calculate_good_score_percentage(df['faithfulness_score']):.1f}%",
                        'Performance Correctness %': f"{calculate_good_score_percentage(df['correctness_score']):.1f}%",
                        'Performance Retrievability %': f"{calculate_good_score_percentage(df['retrievability_score']):.1f}%",
                        'Number of Evaluations': len(df)
                    }
                    stats_data.append(model_stats)
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, height=200)
                
                # FIXME Columns needs to be filtered
                # Combined results with all scores and reasoning
                st.subheader("Detailed Combined Results")
                st.dataframe(combined_df, height=400)
    
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")

if __name__ == "__main__":
    main()