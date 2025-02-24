def create_metrics_summary(experiments_data):
    """Create a summary of good score percentages across metrics and LLMs."""
    import pandas as pd
    import plotly.graph_objects as go
    
    # Get all available metrics and judges
    all_metrics = set()
    all_judges = set()
    
    for exp in experiments_data:
        for judge, df in exp['dfs'].items():
            all_judges.add(judge)
            metrics = [col.replace('_score', '') for col in df.columns if col.endswith('_score')]
            all_metrics.update(metrics)
    
    # Create dataframes for metric averages
    metric_summary = {metric: [] for metric in all_metrics}
    
    # Calculate average good score percentage for each experiment and metric
    for exp in experiments_data:
        exp_name = exp['name']
        
        for metric in sorted(all_metrics):
            judges_values = []
            
            for judge, df in exp['dfs'].items():
                score_col = f'{metric}_score'
                if score_col in df.columns:
                    good_score_pct = (df[score_col][df[score_col].isin([4, 5])].count() / 
                                      df[score_col].count() * 100)
                    judges_values.append(good_score_pct)
            
            if judges_values:
                avg_score = sum(judges_values) / len(judges_values)
                metric_summary[metric].append({
                    'Experiment': exp_name,
                    'Average Good Score %': avg_score,
                    'Judges Count': len(judges_values)
                })
    
    # Create summary dataframes and figures
    summary_dfs = {}
    summary_figs = {}
    
    for metric, data in metric_summary.items():
        if data:
            # Create dataframe
            summary_df = pd.DataFrame(data)
            summary_df['Average Good Score %'] = summary_df['Average Good Score %'].round(1)
            
            # Format highest score in bold by creating a new display column
            max_score = summary_df['Average Good Score %'].max()
            summary_df['Display Score'] = summary_df['Average Good Score %'].apply(
                lambda x: f"**{x:.1f}%**" if x == max_score else f"{x:.1f}%"
            )
            summary_dfs[metric] = summary_df
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=summary_df['Experiment'],
                y=summary_df['Average Good Score %'],
                text=summary_df['Average Good Score %'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker_color='rgb(55, 83, 139)',
                hovertemplate='Experiment: %{x}<br>Average Good Score: %{y:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Average {metric.title()} Good Scores Across Judges",
                xaxis_title="Experiment",
                yaxis_title="Average Good Score %",
                yaxis=dict(
                    ticksuffix="%",
                    range=[0, 100]
                ),
                height=400,
                showlegend=False
            )
            
            summary_figs[metric] = fig
    
    # Create overall average figure
    overall_data = []
    
    for exp in experiments_data:
        exp_name = exp['name']
        metrics_values = {}
        
        for metric in sorted(all_metrics):
            metric_values = []
            
            for judge, df in exp['dfs'].items():
                score_col = f'{metric}_score'
                if score_col in df.columns:
                    good_score_pct = (df[score_col][df[score_col].isin([4, 5])].count() / 
                                      df[score_col].count() * 100)
                    metric_values.append(good_score_pct)
            
            if metric_values:
                metrics_values[metric] = sum(metric_values) / len(metric_values)
        
        if metrics_values:
            overall_avg = sum(metrics_values.values()) / len(metrics_values)
            overall_data.append({
                'Experiment': exp_name,
                'Overall Average': overall_avg,
                **metrics_values
            })
    
    overall_df = pd.DataFrame(overall_data)
    
    # Round values for better display
    for col in overall_df.columns:
        if col != 'Experiment':
            overall_df[col] = overall_df[col].round(1)
    
    # Create overall summary figure
    overall_fig = go.Figure()
    
    # Add a trace for each metric and the overall average
    colors = ['rgb(55, 83, 139)', 'rgb(26, 118, 255)', 'rgb(103, 171, 131)', 'rgb(225, 87, 89)']
    
    for i, metric in enumerate(['Overall Average'] + sorted(all_metrics)):
        if metric in overall_df.columns:
            overall_fig.add_trace(go.Bar(
                x=overall_df['Experiment'],
                y=overall_df[metric],
                text=overall_df[metric].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                name=metric.title(),
                marker_color=colors[i % len(colors)],
                hovertemplate='Experiment: %{x}<br>' + metric.title() + ': %{y:.1f}%<extra></extra>'
            ))
    
    overall_fig.update_layout(
        title="Overall Performance Summary",
        xaxis_title="Experiment",
        yaxis_title="Good Score %",
        yaxis=dict(
            ticksuffix="%",
            range=[0, 100]
        ),
        height=500,
        barmode='group',
        showlegend=True,
        legend=dict(
            title="Metrics",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return summary_dfs, summary_figs, overall_df, overall_fig