"""Visualization components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors

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
def create_radar_plot(experiments_data):
    """Create a radar plot comparing performance metrics across experiments."""

    # Get all available metrics and judges
    all_metrics = set()
    all_judges = set()
    for exp in experiments_data:
        for judge, df in exp['dfs'].items():
            all_judges.add(judge)
            metrics = [col.replace('_score', '') for col in df.columns if col.endswith('_score')]
            all_metrics.update(metrics)
    
    # Sort metrics alphabetically for consistent order
    sorted_metrics = sorted(all_metrics)
    
    # Use vibrant color schemes that are more visually distinct
    color_schemes = [
        plotly.colors.qualitative.Plotly,     # Bright default Plotly colors
        plotly.colors.qualitative.D3,         # D3 color scheme
        plotly.colors.qualitative.G10,        # G10 color scheme
        plotly.colors.qualitative.T10         # Tableau 10 color scheme
    ]
    
    # Flatten the color schemes to have one large color pool
    all_colors = []
    for scheme in color_schemes:
        all_colors.extend(scheme)
    
    fig = go.Figure()
    
    # Create traces for each experiment and judge
    for exp_idx, exp in enumerate(experiments_data):
        for judge in all_judges:
            if judge in exp['dfs']:
                df = exp['dfs'][judge]
                metrics_values = []
                
                # Calculate good score percentages for each metric
                for metric in sorted_metrics:
                    score_col = f'{metric}_score'
                    if score_col in df.columns:
                        good_score_pct = (df[score_col][df[score_col].isin([4, 5])].count() / 
                                        df[score_col].count() * 100)
                        metrics_values.append(good_score_pct)
                    else:
                        metrics_values.append(0)
                
                # IMPORTANT: Make the shape close by repeating the first point at the end
                # First, get the list of metrics and values
                display_metrics = sorted_metrics.copy()
                metrics_values_closed = metrics_values.copy()
                
                # Add the first metric and its value to the end to create a closed loop
                # Only needed when connecting manually, not using fill='toself'
                display_metrics.append(display_metrics[0])  
                metrics_values_closed.append(metrics_values_closed[0])
                
                # Generate a distinct color index for each experiment-judge combination
                color_idx = (exp_idx * len(all_judges) + list(all_judges).index(judge)) % len(all_colors)
                color = all_colors[color_idx]
                
                # Add trace with higher line opacity, clear fill, and larger markers
                fig.add_trace(go.Scatterpolar(
                    r=metrics_values,
                    theta=sorted_metrics,
                    name=f"{exp['name']} - {judge}",
                    line=dict(
                        color=color,
                        width=3          # Thicker lines for better visibility
                    ),
                    fill='toself',       # Fill the area inside the radar
                    fillcolor=color,
                    opacity=0.4,         # Semi-transparent fill
                    marker=dict(
                        size=10,         # Larger markers for better visibility
                        color=color,
                        line=dict(
                            width=2,
                            color='white'  # White border around markers for contrast
                        )
                    ),
                    showlegend=True
                ))
    
    # Improve radar plot appearance
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                showticklabels=True,
                ticks='outside',
                gridcolor='rgba(200, 200, 200, 1)',  # Lighter grid
                linecolor='rgba(200, 200, 200, 1)'   # Clear axis line
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='black'),  # Make metric labels more readable
                rotation=90,                             # Start from the top
                direction='clockwise'                   # Standard radar plot direction
            ),
            bgcolor='rgba(240, 240, 240, 0.1)'          # Light background color
        ),
        showlegend=True,
        title=dict(
            text="Performance Metrics Radar Plot (% of Good Scores)",
            font=dict(size=20)
        ),
        height=600,
        width=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=80, r=80, t=100, b=80)  # Adjust margins for better fit
    )
    
    return fig