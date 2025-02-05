"""Visualization components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px

def create_timing_plot(experiments_data, column_name, title):
    """Create timing distribution plot."""
    fig = go.Figure()
    for exp in experiments_data:
        if column_name in exp['timing'].columns:
            fig.add_trace(go.Box(
                y=exp['timing'][column_name],
                name=exp['name'],
                boxpoints='all'
            ))
    fig.update_layout(
        title=title,
        height=500,
        yaxis_title="Time (seconds)"
    )
    return fig

def create_score_distribution_plot(experiments_data, metric, plot_type="violin"):
    """Create score distribution plot."""
    fig = go.Figure()
    
    for exp_idx, exp in enumerate(experiments_data):
        colors = px.colors.qualitative.Set2 if exp_idx == 0 else px.colors.qualitative.Set3
        
        for judge_idx, (judge, df) in enumerate(exp['dfs'].items()):
            metric_col = f"{metric}_score"
            if metric_col in df.columns:
                if plot_type == "violin":
                    fig.add_trace(go.Violin(
                        y=df[metric_col],
                        name=f"{exp['name']} - {judge}",
                        box_visible=True,
                        meanline_visible=True,
                        points='all',
                        line_color=colors[judge_idx],
                        side='positive' if exp_idx == 0 else 'negative',
                        showlegend=True
                    ))
                else:  # box plot
                    fig.add_trace(go.Box(
                        y=df[metric_col],
                        name=f"{exp['name']} - {judge}",
                        boxpoints='outliers',
                        marker_color=colors[judge_idx],
                        showlegend=True
                    ))
    
    fig.update_layout(
        title=f"{metric.title()} Score Distribution Analysis",
        yaxis_title="Score",
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 5.5]
        ),
        height=600,
        showlegend=True,
        legend=dict(
            title="Jury Evaluation",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    if plot_type == "violin":
        fig.update_layout(
            violingap=0.1,
            violingroupgap=0
        )
    else:
        fig.update_layout(
            boxgap=0.1,
            boxgroupgap=0.1
        )
    
    return fig