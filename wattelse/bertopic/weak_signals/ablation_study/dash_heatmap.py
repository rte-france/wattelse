import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Load the data
folder = "signal_evolution_data/"  # Set your folder path
metadata = pd.read_pickle(folder + 'metadata.pkl')
noise_df_list = pd.read_pickle(folder + 'noise_dfs_over_time.pkl')
ws_df_list = pd.read_pickle(folder + 'weak_signal_dfs_over_time.pkl')
ss_df_list = pd.read_pickle(folder + 'strong_signal_dfs_over_time.pkl')

# Gather all the unique topic IDs and their representations from all dataframes at all timestamps
topic_representations = {}

for df_list in [noise_df_list, ws_df_list, ss_df_list]:
    for df in df_list:
        if not df.empty:
            for _, row in df.iterrows():
                topic_number = row['Topic']
                topic_number_str = str(topic_number)
                topic_representations[topic_number_str] = f"{topic_number_str}_{row['Representation']}"

# Sort topics by their number
sorted_topic_representations = dict(sorted(topic_representations.items(), key=lambda item: int(item[0])))

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'backgroundColor': '#111111',
        'color': '#ffffff',
        'fontFamily': 'Arial, sans-serif',
        'padding': '20px'
    },
    children=[
        html.H1('Topic Visualization Dashboard', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Label('Search Keywords:', style={'display': 'block', 'marginBottom': '10px'}),
            dcc.Input(id='search-input', type='text', placeholder='Enter keywords (comma-separated)', style={'width': '100%', 'padding': '10px'}),
        ], style={'marginBottom': '20px'}),
        html.H2('Search Results', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div(id='output-graph', style={'width': '100%'})
    ]
)

@app.callback(
    Output('output-graph', 'children'),
    [Input('search-input', 'value')]
)
def update_graph(search_keywords):
    if not search_keywords:
        return []

    search_keywords = [keyword.strip().lower() for keyword in search_keywords.split(',')]

    # Filter topics based on search keywords
    filtered_topic_representations = {topic: representation for topic, representation in sorted_topic_representations.items()
                                      if all(keyword in representation.lower() for keyword in search_keywords)}

    charts = []

    for topic, representation in filtered_topic_representations.items():
        heatmap_data = []
        popularity_data = []

        for t in range(len(metadata['timestamps'])):
            ts = pd.to_datetime(metadata['timestamps'][t]).strftime('%Y-%m')
            
            signal_strength = 0
            popularity = 0
            
            if not noise_df_list[t].empty and int(topic) in noise_df_list[t]['Topic'].tolist():
                signal_strength = 1
                popularity = noise_df_list[t].loc[noise_df_list[t]['Topic'] == int(topic), 'Latest Popularity'].values[0]
            elif not ws_df_list[t].empty and int(topic) in ws_df_list[t]['Topic'].tolist():
                signal_strength = 2
                popularity = ws_df_list[t].loc[ws_df_list[t]['Topic'] == int(topic), 'Latest Popularity'].values[0]
            elif not ss_df_list[t].empty and int(topic) in ss_df_list[t]['Topic'].tolist():
                signal_strength = 3
                popularity = ss_df_list[t].loc[ss_df_list[t]['Topic'] == int(topic), 'Latest Popularity'].values[0]
            
            heatmap_data.append(signal_strength)
            popularity_data.append(popularity)

        # Create subplots for heatmap and popularity plot
        fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], vertical_spacing=0.1)

        # Add heatmap trace
        fig.add_trace(go.Heatmap(
            z=[heatmap_data],
            x=metadata['timestamps'],
            y=[representation],
            colorscale=[[0, 'white'], [0.33, 'blue'], [0.66, 'orange'], [1, 'red']],
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate='Timestamp: %{x}<br>Signal Strength: %{z}<extra></extra>'
        ), row=1, col=1)

        # Add popularity plot trace
        fig.add_trace(go.Scatter(
            x=metadata['timestamps'],
            y=popularity_data,
            mode='lines',
            name=representation,
            hovertemplate='Timestamp: %{x}<br>Popularity: %{y}<extra></extra>',
            line=dict(color='#ffffff')
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'Topic: {representation}',
            xaxis=dict(title='Timestamp'),
            yaxis=dict(title='Signal Strength'),
            xaxis2=dict(title='Timestamp'),
            yaxis2=dict(title='Popularity'),
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='#ffffff'),
            height=600,
            width=1200,  # Set width to a numeric value
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False,
            hovermode='x unified'
        )

        charts.append(html.Div(
            style={'marginBottom': '40px', 'width': '100%'},
            children=dcc.Graph(figure=fig)
        ))

    return charts

if __name__ == '__main__':
    app.run_server(debug=True)
