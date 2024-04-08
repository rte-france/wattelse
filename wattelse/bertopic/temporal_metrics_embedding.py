from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from bertopic import BERTopic
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
import itertools
import plotly.graph_objects as go
import plotly.express as px
import torch
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from loguru import logger

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import plotly.graph_objs as go



class TempTopic:
    
    def __init__(self, topic_model: BERTopic, 
                 docs: List[str], 
                 embeddings: List[List[float]],
                 word_embeddings: List[List[List[float]]],
                 token_strings: List[List[str]],
                 timestamps: Union[List[str], List[int]], 
                 topics: List[int] = None,
                 evolution_tuning: bool = True,
                 global_tuning: bool = False):
        """
        Initializes the TempTopic object with a BERTopic model, a list of documents, 
        their corresponding timestamps, and optionally a list of topics.

        Parameters:
        - topic_model: A trained BERTopic model.
        - docs: A list of documents (strings).
        - timestamps: A list of timestamps corresponding to each document. The list can contain strings or integers.
        - topics: An optional list of topics corresponding to each document.
        - evolution_tuning: Boolean to fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF at t-1.
        - global_tuning: Boolean indicating whether to apply global tuning to align topics with the global c-TF-IDF representation.

        Raises:
        - ValueError: If the lengths of docs, timestamps, and optionally topics are not the same.
        - TypeError: If the provided arguments are not of the expected types.
        """
        if not isinstance(topic_model, BERTopic):
            raise TypeError("topic_model must be an instance of BERTopic.")
        if not isinstance(docs, list) or not all(isinstance(doc, str) for doc in docs):
            raise TypeError("docs must be a list of strings.")
        if not isinstance(timestamps, list) or not all(isinstance(t, (str, int, float)) for t in timestamps):
            raise TypeError("timestamps must be a list of str, int or float.")
        if topics is not None and (not isinstance(topics, list) or not all(isinstance(topic, int) for topic in topics)):
            raise TypeError("topics, if provided, must be a list of integers.")

        # Ensure all inputs have the same length
        if topics is not None and not (len(docs) == len(timestamps) == len(topics)):
            raise ValueError("Lengths of docs, timestamps, and topics must all be the same.")
        elif not (len(docs) == len(timestamps)):
            raise ValueError("Lengths of docs and timestamps must be the same.")

        self.topic_model = topic_model
        self.docs = docs
        self.embeddings = embeddings
        self.word_embeddings = word_embeddings
        self.token_strings = token_strings
        self.timestamps = timestamps
        self.topics = topics if topics is not None else self.topic_model.topics_
        self.global_tuning = global_tuning
        self.evolution_tuning = evolution_tuning
        
        self.final_df = None
        self.representation_embeddings_df = None


        self.stability_scores_df = None
        self.representation_stability_scores_df = None
        self.overall_stability_df = None
        self.avg_stability_score, self.avg_representation_stability_score = None, None

    def _topics_over_time(self) -> pd.DataFrame:
        """
        Extends the existing method to include document embeddings and the mean topic embedding for each topic at each timestamp.
        """

        doc_embeddings = list(map(lambda x: x, self.embeddings))
        token_embeddings = list(map(lambda x: x, self.word_embeddings))
        token_strings = list(map(lambda x: x, self.token_strings))
        # print(len(self.docs), len(self.timestamps), len(self.topics), len(doc_embeddings), len(token_embeddings), len(token_strings))


        # Prepare documents DataFrame
        documents = pd.DataFrame({
            "Document": self.docs,
            "Timestamps": self.timestamps,
            "Topic": self.topics,
            "Document_Embeddings": doc_embeddings,
            "Token_Embeddings": token_embeddings,
            "Token_Strings": token_strings
        })

        # Remove rows associated with outlier topic
        documents = documents[documents['Topic'] != -1]

        # Normalize the global c-TF-IDF representation for tuning purposes
        global_c_tf_idf = normalize(self.topic_model.c_tf_idf_, axis=1, norm='l1', copy=False)

        # Ensure all topics are processed in order
        all_topics = sorted(list(documents.Topic.unique()))
        all_topics_indices = {topic: index for index, topic in enumerate(all_topics)}

        # Sort documents by their timestamps for sequential processing
        documents.sort_values("Timestamps", inplace=True)
        timestamps = documents["Timestamps"].unique()

        topics_over_time = []  # Accumulates the final data for each timestamp
        document_per_topic_list = []  # Tracks documents associated with each topic at each timestamp

        for index, timestamp in tqdm(enumerate(timestamps), desc="Initial processing"):
            # Select documents for the current timestamp
            selection = documents.loc[documents.Timestamps == timestamp, :]
            
            # Aggregate documents by topic to compute c-TF-IDF and collect embeddings
            documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({
                'Document': ' '.join,  # Combine documents for each topic
                "Timestamps": "count",  # Count of documents per topic
                "Document_Embeddings": lambda x: list(x),  # Collect embeddings in a list
                "Token_Embeddings": lambda x: list(itertools.chain.from_iterable(x)),  # Flatten and collect token embeddings
                "Token_Strings": lambda x: list(itertools.chain.from_iterable(x))  # Flatten and collect token strings
            })

            # Compute c-TF-IDF for the current selection
            c_tf_idf, words = self.topic_model._c_tf_idf(documents_per_topic, fit=False)

            documents_per_topic_2 = selection.groupby('Topic', as_index=False).agg({
                'Document': lambda docs: list(docs),
                'Document_Embeddings': lambda x: list(x),  # Collect embeddings in a list
                'Token_Embeddings': lambda x: list(x),  # Collect token embeddings in a list
                'Token_Strings': lambda x: list(x)  # Collect token strings in a list
            })
            documents_per_topic_2['Timestamp'] = timestamp

            # Calculate mean embeddings for each topic
            documents_per_topic_2['Embedding'] = documents_per_topic_2['Document_Embeddings'].apply(lambda x: np.mean(x, axis=0))

            document_per_topic_list.append(documents_per_topic_2)

            # Evolution tuning and global tuning logic remains the same
            if self.evolution_tuning and index != 0:
                current_topics = sorted(list(documents_per_topic.Topic.values))
                overlapping_topics = sorted(list(set(previous_topics).intersection(set(current_topics))))

                current_overlap_idx = [current_topics.index(topic) for topic in overlapping_topics]
                previous_overlap_idx = [previous_topics.index(topic) for topic in overlapping_topics]

                c_tf_idf.tolil()[current_overlap_idx] = ((c_tf_idf[current_overlap_idx] +
                                                            previous_c_tf_idf[previous_overlap_idx]) / 2.0).tolil()

            # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
            if self.global_tuning:
                selected_topics = [all_topics_indices[topic] for topic in documents_per_topic.Topic.values]
                c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0

            # Extract the words per topic
            words_per_topic = self.topic_model._extract_words_per_topic(words, selection, c_tf_idf, calculate_aspects=False)
            topic_frequency = pd.Series(documents_per_topic.Timestamps.values,
                                        index=documents_per_topic.Topic).to_dict()

            # Fill dataframe with results, now including document embeddings and the mean topic embedding
            topics_at_timestamp = [(topic,
                        ", ".join([words[0] for words in values][:5]),
                        topic_frequency[topic],
                        timestamp,
                        documents_per_topic_2.loc[documents_per_topic_2['Topic'] == topic, 'Document_Embeddings'].values[0],
                        documents_per_topic_2.loc[documents_per_topic_2['Topic'] == topic, 'Embedding'].values[0],
                        documents_per_topic_2.loc[documents_per_topic_2['Topic'] == topic, 'Token_Embeddings'].values[0],
                        documents_per_topic_2.loc[documents_per_topic_2['Topic'] == topic, 'Token_Strings'].values[0])
                    for topic, values in words_per_topic.items()]
            
            topics_over_time.extend(topics_at_timestamp)
            
            if self.evolution_tuning:
                previous_topics = sorted(list(documents_per_topic.Topic.values))
                previous_c_tf_idf = c_tf_idf.copy()

        # Adjust the DataFrame creation to include new columns for document embeddings and mean topic embeddings
        columns = ["Topic", "Words", "Frequency", "Timestamp", "Document_Embeddings", "Embedding", "Token_Embeddings", "Token_Strings"]
        topics_over_time_df = pd.DataFrame(topics_over_time, columns=columns)
        topics_over_time_df['Word_Count'] = topics_over_time_df['Words'].apply(lambda x: len(x.split(', ')))
        topics_over_time_df = topics_over_time_df[topics_over_time_df['Word_Count'] == 5]
        self.final_df = topics_over_time_df.merge(pd.concat(document_per_topic_list), on=['Topic', 'Timestamp'], how='left', suffixes=('', '_drop'))

        # Drop duplicate columns resulted from merge
        self.final_df.drop([col for col in self.final_df.columns if 'drop' in col], axis=1, inplace=True)
    
    
    def fit(self):
        self._topics_over_time()
        self._calculate_representation_embeddings()
    

    ###################### NEW METRICS BASED ON EMBEDDINGS ######################


    def calculate_temporal_stability(self, window_size: int = 2) -> Tuple[pd.DataFrame, float]:
        """
        Calculates the Temporal Topic Stability scores for topics across different timestamps using cosine similarity
        of topic embeddings. This method provides insights into the stability of topics over time, including the topic's
        word representation at the start and end of the specified window.

        Parameters:
        - window_size (int): Specifies the temporal window size for calculating stability. Must be 2 or above.

        Returns:
        - Tuple[pd.DataFrame, float]: A tuple containing a DataFrame and a float value.
        The DataFrame includes columns for 'Topic ID', 'Start Timestamp', 'End Timestamp', 'Start Topic', 'End Topic', 
        and 'Stability Score', detailing the stability scores for topic embeddings between timestamps. The float value 
        represents the average Stability score across all topics and timestamps, providing a measure of overall temporal 
        topic stability in the dataset.

        Raises:
        - ValueError: If window_size is less than 2.
        """

        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")
        
        if self.final_df is None:
            self._topics_over_time()

        # Initialize DataFrame for Stability scores
        self.stability_scores_df = pd.DataFrame(columns=['Topic ID', 'Start Timestamp', 'End Timestamp', 'Start Topic', 'End Topic', 'Stability Score'])

        grouped_topics = self.final_df.groupby('Topic')
        scores_list = []

        for topic_id, group in grouped_topics:
            sorted_group = group.sort_values('Timestamp')

            for i in range(len(sorted_group) - window_size + 1):
                start_row = sorted_group.iloc[i]
                end_row = sorted_group.iloc[i + window_size - 1]

                start_embedding = np.array(start_row['Embedding']).reshape(1, -1)
                end_embedding = np.array(end_row['Embedding']).reshape(1, -1)

                # Calculate cosine similarity
                similarity_score = cosine_similarity(start_embedding, end_embedding)[0][0]

                scores_list.append({
                    'Topic ID': topic_id,
                    'Start Timestamp': start_row['Timestamp'],
                    'End Timestamp': end_row['Timestamp'],
                    'Start Topic': start_row['Words'],  # Words representation at the start of the window
                    'End Topic': end_row['Words'],  # Words representation at the end of the window
                    'Stability Score': similarity_score
                })

        self.stability_scores_df = pd.DataFrame(scores_list)
        self.avg_stability_score = self.stability_scores_df['Stability Score'].mean()

        return self.stability_scores_df, self.avg_stability_score
    

    def _calculate_representation_embeddings(self, double_agg: bool = True, doc_agg: str = "mean", global_agg: str = "max"):
        """
        Calculate the aggregated word embeddings for each topic's representation at each timestamp.
        """
        representation_embeddings = []

        for _, row in self.final_df.iterrows():
            topic_id = row['Topic']
            timestamp = row['Timestamp']
            representation = [word.lower() for word in row['Words'].split(', ')]
            token_strings = [[token.lower() for token in tokens] for tokens in row['Token_Strings']]
            word_embeddings = row['Token_Embeddings']

            embedding_list = []
            updated_representation = []  # Store the updated representation without missing words
            for word in representation:
                if word != '':
                    try:
                        embedding = self._get_aggregated_word_embedding(word, word_embeddings, token_strings, double_agg, doc_agg, global_agg)
                        embedding_list.append(embedding)
                        updated_representation.append(word)  # Add the word to the updated representation
                    except ValueError:
                        logger.debug(f"Word '{word}' not found in topic {topic_id} at timestamp {timestamp}. Retrying by looking inside the entire topic.")
                        topic_data = self.final_df[self.final_df['Topic'] == topic_id]
                        all_token_strings = list(itertools.chain.from_iterable(topic_data['Token_Strings'].tolist()))
                        all_token_strings = [[token.lower() for token in doc_tokens] for doc_tokens in all_token_strings]
                        all_word_embeddings = list(itertools.chain.from_iterable(topic_data['Token_Embeddings'].tolist()))

                        try:
                            embedding = self._get_aggregated_word_embedding(word, all_word_embeddings, all_token_strings, double_agg, doc_agg, global_agg)
                            embedding_list.append(embedding)
                            updated_representation.append(word)  # Add the word to the updated representation
                        except ValueError:
                            logger.debug(f"The word '{word}' was not found in any of the documents for topic {topic_id}. Skipping...")
                            # Skip the word and continue without adding it to the updated representation

            representation_embeddings.append({
                'Topic ID': topic_id,
                'Timestamp': timestamp,
                'Representation': ', '.join(updated_representation),  # Use the updated representation
                'Representation Embeddings': embedding_list
            })

        self.representation_embeddings_df = pd.DataFrame(representation_embeddings)


    def _get_word_embeddings(self, token_strings, word_embeddings, word):
        """
        Get embeddings for a word from the token strings and word embeddings.
        Returns a list of embeddings, where each embedding corresponds to a document.
        """
        all_embeddings = []
        for i, tokens in enumerate(token_strings):
            if word in tokens:
                indices = [j for j, token in enumerate(tokens) if token == word]
                embeddings = [word_embeddings[i][j] for j in indices]
                all_embeddings.append(embeddings)
        return all_embeddings

    def _aggregate_embeddings(self, embeddings_array, agg_type='mean'):
        """
        Aggregate embeddings using the specified method.
        """
        if agg_type == 'mean':
            return np.mean(embeddings_array, axis=0)
        elif agg_type == 'max':
            return np.max(embeddings_array, axis=0)
        elif agg_type == 'min':
            return np.min(embeddings_array, axis=0)
        else:
            raise ValueError("Unsupported aggregation type.")

    def _get_aggregated_word_embedding(self, word, word_embeddings, token_strings, double_agg=True, doc_agg='mean', global_agg='max'):
        """
        Compute an aggregated word embedding.
        """
        all_embeddings = self._get_word_embeddings(token_strings, word_embeddings, word)
        if not all_embeddings:
            raise ValueError(f"The word '{word}' was not found in the provided token strings.")
        
        if double_agg:
            # Perform document-level aggregation first, if specified
            doc_embeddings = []
            for doc_embedding_list in all_embeddings:
                if len(doc_embedding_list) == 1:
                    doc_embeddings.append(doc_embedding_list[0])
                else:
                    doc_embeddings.append(self._aggregate_embeddings(np.stack(doc_embedding_list), doc_agg))
            
            if len(doc_embeddings) == 1:
                final_embedding = doc_embeddings[0]
            else:
                final_embedding = self._aggregate_embeddings(np.stack(doc_embeddings), global_agg)
        else:
            # Directly apply the specified aggregation across all word occurrences
            flat_embeddings = [embedding for doc_list in all_embeddings for embedding in doc_list]
            if len(flat_embeddings) == 1:
                final_embedding = flat_embeddings[0]
            else:
                final_embedding = self._aggregate_embeddings(np.stack(flat_embeddings), doc_agg)
        
        return final_embedding



    def calculate_temporal_representation_stability(self, 
                                                    window_size: int = 2, 
                                                    k: int = 1,
                                                    double_agg: bool = True,
                                                    doc_agg: str = "mean",
                                                    global_agg: str ="max") -> Tuple[pd.DataFrame, float]:
        """
        Calculates the Temporal Topic Representation Stability scores for topics across different timestamps using cosine similarity
        of word embeddings in the topic's representation. This method provides insights into the stability of topic representations
        over time by comparing the word embeddings of a topic's representation at different timestamps.

        Parameters:
        - window_size (int): Specifies the temporal window size for calculating stability. Must be 2 or above.
        - k (int): Specifies the number of nearest embeddings to consider when comparing word embeddings. Default is 1.

        Returns:
        - Tuple[pd.DataFrame, float]: A tuple containing a DataFrame and a float value.
        The DataFrame includes columns for 'Topic ID', 'Start Timestamp', 'End Timestamp', 'Start Representation', 'End Representation',
        and 'Representation Stability Score', detailing the stability scores for topic representations between timestamps. The float
        value represents the average Representation Stability score across all topics and timestamps, providing a measure of overall
        temporal topic representation stability in the dataset.

        Raises:
        - ValueError: If window_size is less than 2.
        """

        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")

        self._calculate_representation_embeddings(double_agg, doc_agg, global_agg)

        stability_scores = []

        grouped_topics = self.representation_embeddings_df.groupby('Topic ID')
        for topic_id, group in grouped_topics:
            sorted_group = group.sort_values('Timestamp')

            for i in range(len(sorted_group) - window_size + 1):
                start_row = sorted_group.iloc[i]
                end_row = sorted_group.iloc[i + window_size - 1]

                start_embeddings = start_row['Representation Embeddings']
                end_embeddings = end_row['Representation Embeddings']

                similarity_scores = []
                for start_embedding in start_embeddings:
                    cosine_similarities = cosine_similarity([start_embedding], end_embeddings)[0]
                    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
                    top_k_similarities = cosine_similarities[top_k_indices]
                    similarity_scores.extend(top_k_similarities)

                avg_similarity = np.mean(similarity_scores)

                stability_scores.append({
                    'Topic ID': topic_id,
                    'Start Timestamp': start_row['Timestamp'],
                    'End Timestamp': end_row['Timestamp'],
                    'Start Representation': start_row['Representation'],
                    'End Representation': end_row['Representation'],
                    'Representation Stability Score': avg_similarity
                })

        self.representation_stability_scores_df = pd.DataFrame(stability_scores)
        self.avg_representation_stability_score = self.representation_stability_scores_df['Representation Stability Score'].mean()

        return self.representation_stability_scores_df, self.avg_representation_stability_score
    


    def calculate_overall_topic_stability(self, 
                                          window_size: int = 2, 
                                          k: int = 1, 
                                          alpha: float = 0.5,
                                          double_agg: bool = True,
                                          doc_agg: str = "mean",
                                          global_agg: str = "max") -> pd.DataFrame:
        """
        Calculates the Overall Topic Stability scores for each topic by combining the Temporal Topic Stability and
        Temporal Topic Representation Stability scores. The Overall Topic Stability score is a weighted average of
        the two stability scores, where the weight is determined by the alpha parameter.

        Parameters:
        - window_size (int): Specifies the temporal window size for calculating stability. Must be 2 or above.
        - k (int): Specifies the number of nearest embeddings to consider when comparing word embeddings. Default is 1.
        - alpha (float): Specifies the weight given to the Temporal Topic Stability score. Must be between 0 and 1.
                         Default is 0.5, giving equal weight to both stability scores.

        Returns:
        - pd.DataFrame: A DataFrame containing the Overall Topic Stability scores for each topic. The DataFrame has columns
                        for 'Topic ID' and 'Overall Topic Stability Score'.

        Raises:
        - ValueError: If window_size is less than 2 or if alpha is not between 0 and 1.
        """

        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1.")

        # Calculate Temporal Topic Stability scores
        topic_stability_df, _ = self.calculate_temporal_stability(window_size)

        # Calculate Temporal Topic Representation Stability scores
        representation_stability_df, _ = self.calculate_temporal_representation_stability(window_size, k, double_agg, doc_agg, global_agg)

        # Merge the two DataFrames on 'Topic ID', 'Start Timestamp', and 'End Timestamp'
        merged_df = pd.merge(topic_stability_df, representation_stability_df,
                             on=['Topic ID', 'Start Timestamp', 'End Timestamp'])

        # Calculate the weighted average of the two stability scores for each topic and timestamp
        merged_df['Overall Stability Score'] = alpha * merged_df['Stability Score'] + \
                                               (1 - alpha) * merged_df['Representation Stability Score']

        # Group by 'Topic ID' and calculate the average Overall Stability Score for each topic
        self.overall_stability_df = merged_df.groupby('Topic ID')['Overall Stability Score'].mean().reset_index()

        return self.overall_stability_df
    


    def plot_overall_topic_stability(self, darkmode: bool = True, topics_to_show: List[int] = None):
        """
        Plot overall topic stability as a histogram with a color gradient from blue to red.
        Green: High overall topic stability
        Red: Low overall topic stability

        Parameters:
        - darkmode: for the aesthetic of the plot only
        - topics_to_show: Optional list of topic IDs to display. If None or empty, show all.
        """

        if self.overall_stability_df is None:
            self.calculate_overall_topic_stability()
            

        metric_column = "Overall Stability Score"
        df = self.overall_stability_df

        # If topics_to_show is None or empty, include all topics; otherwise, filter
        if topics_to_show is None or len(topics_to_show) == 0:
            topics_to_show = df['Topic ID'].unique()
        df = df[df['Topic ID'].isin(topics_to_show)]

        # Order by Topic ID for x-axis display
        df = df.sort_values(by='Topic ID')

        # Normalize Overall Topic Stability Score for color mapping
        df['ScoreNormalized'] = (df[metric_column] - df[metric_column].min()) / (df[metric_column].max() - df[metric_column].min())
        df['Color'] = df['ScoreNormalized'].apply(lambda x: px.colors.diverging.RdYlGn[int(x * (len(px.colors.diverging.RdYlGn) - 1))])

        # Initialize a Plotly figure with a dark theme
        if darkmode:
            fig = go.Figure(layout=go.Layout(template="plotly_dark"))
        else:
            fig = go.Figure()

        # Create a histogram (bar chart) for the overall topic stability scores
        for _, row in df.iterrows():
            topic_id = row['Topic ID']
            metric_value = row[metric_column]
            words = self.final_df[self.final_df['Topic'] == topic_id]['Words'].values[0] if topic_id in self.final_df['Topic'].values else "No words"
            
            fig.add_trace(go.Bar(
                x=[topic_id],
                y=[metric_value],
                marker_color=row['Color'],
                name=f'Topic {topic_id}',
                hovertext=f"Topic {topic_id} Representation: {words}",
                hoverinfo="text+y"
            ))

        # Update layout for histogram
        fig.update_layout(
            title='Overall Topic Stability Scores',
            xaxis=dict(title='Topic ID'),
            yaxis=dict(title='Overall Topic Stability Score'),
            showlegend=False
        )

        return fig


    def plot_temporal_stability_metrics(self, metric: str, darkmode: bool = True, topics_to_show: List[int] = None, smoothing_factor: float = 0.2):
        """
        Plot temporal topic stability or temporal topic representation stability with an option to select specific topics.

        Parameters:
        - metric: The metric to plot ('topic_stability' or 'representation_stability').
        - darkmode: for the aesthetic of the plot only
        - topics_to_show: Optional list of topic IDs to display. If None or empty, show all.
        - smoothing_factor: The factor used for smoothing the plot lines. Default is 0.1.
        """
        # Initialize a Plotly figure
        if darkmode:
            fig = go.Figure(layout=go.Layout(template="plotly_dark"))
        else:
            fig = go.Figure()

        # Determine which topics to plot
        if topics_to_show is None or not topics_to_show:
            topics_to_show = self.final_df['Topic'].unique()

        # Determine the DataFrame and score column based on the selected metric
        if metric == 'topic_stability':
            df = self.stability_scores_df
            score_column = 'Stability Score'
            rep_column = 'Start Topic'
        elif metric == 'representation_stability':
            df = self.representation_stability_scores_df
            score_column = 'Representation Stability Score'
            rep_column = 'Start Representation'
        else:
            raise ValueError("Invalid metric. Please choose either 'topic_stability' or 'representation_stability'.")

        topics_to_show = sorted(topics_to_show)

        # Plot each topic ID for the selected metric
        for topic_id in topics_to_show:
            # Filter data for the current topic ID
            topic_data = df[df['Topic ID'] == topic_id]

            # Sort the data by 'Start Timestamp' to ensure chronological order
            topic_data = topic_data.sort_values(by='Start Timestamp')

            # Get topic words, split them, take the first three, and format them
            topic_words_str = self.final_df[self.final_df['Topic'] == topic_id]['Words'].iloc[0]
            topic_words_list = topic_words_str.split(', ')[:3]  # Split and take first three
            topic_words = "_".join(topic_words_list)

            hover_texts = [f"{words}" for words in topic_data[rep_column]]

            # Apply smoothing to the plot data
            x = topic_data['Start Timestamp']
            y = topic_data[score_column].rolling(window=2, min_periods=1, center=True).mean()

            # Add a scatter plot for the current topic ID with custom label and initially deactivated
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=f'{topic_id}_{topic_words}',
                text=hover_texts,
                hoverinfo='text+x+y',
                visible='legendonly',  # Initially deactivated in legend
                line=dict(shape='spline', smoothing=smoothing_factor)
            ))

        # Update layout
        fig.update_layout(
            title=f'TEMPTopic {metric.replace("_", " ").capitalize()}',
            xaxis_title='Timestamp',
            yaxis_title=f'{metric.replace("_", " ").capitalize()} Score',
            legend_title='Topic',
            hovermode='closest',
        )

        # Show the figure
        return fig


    def _convert_week_to_datetime(self, week_string):
        year, week = week_string.split('-')
        return pd.to_datetime(f"{int(year)}-01-01", format="%Y-%m-%d") + pd.Timedelta(weeks=int(week) - 1)

    def plot_topic_evolution(self, granularity : str, topics_to_show: List[int] = None, perplexity: float = 30.0, learning_rate: float = 200.0, metric: str = 'cosine', color_palette='Plotly'):
        """
        Visualize the evolution of topics in the semantic space using t-SNE dimensionality reduction.

        Parameters:
        - topics_to_show: Optional list of topic IDs to display. If None or empty, show all.
        - granularity: Show evolution by Day, Week, Month or Year.
        - perplexity: Perplexity parameter for t-SNE. Typically between 5 and 50.
        - learning_rate: Learning rate for t-SNE. Typically between 10 and 1000.
        - metric: Metric to use for distance computation in t-SNE ('cosine', 'euclidean', 'manhattan', etc.).
        - color_palette: Color palette to use for topics. Can be 'Plotly', 'D3', 'Alphabet', or a list of colors.
        """

        
        # Get topic information from the topic model
        topic_info = self.topic_model.get_topic_info()
        
        # Create a dictionary to store topic embeddings, timestamps, and words
        topic_data = {}
        for topic_id in self.final_df['Topic'].unique():
            topic_df = self.final_df[self.final_df['Topic'] == topic_id]
            timestamps = topic_df['Timestamp'].apply(self._convert_week_to_datetime if granularity == "Week" else pd.to_datetime)
            topic_data[topic_id] = {
                'embeddings': topic_df['Embedding'].tolist(),
                'timestamps': timestamps,
                'words': topic_df['Words'].tolist()
            }

        # Perform t-SNE dimensionality reduction for all topics
        all_embeddings = np.concatenate([data['embeddings'] for data in topic_data.values()])
        n_samples = all_embeddings.shape[0]
        tsne_perplexity = min(perplexity, n_samples - 1)
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=learning_rate, metric=metric, random_state=42)
        all_embeddings_tsne = tsne.fit_transform(all_embeddings)

        # Split the t-SNE embeddings back into individual topics
        start_idx = 0
        for topic_id, data in topic_data.items():
            end_idx = start_idx + len(data['embeddings'])
            data['embeddings'] = all_embeddings_tsne[start_idx:end_idx]
            start_idx = end_idx

        # Determine which topics to plot
        if topics_to_show is None or not topics_to_show:
            topics_to_show = self.final_df['Topic'].unique()

        # Calculate the overall minimum and maximum values for the axes
        x_min, x_max = all_embeddings_tsne[:, 0].min(), all_embeddings_tsne[:, 0].max()
        y_min, y_max = all_embeddings_tsne[:, 1].min(), all_embeddings_tsne[:, 1].max()
        
        # Convert timestamps to numeric representation based on the selected granularity
        timestamps_numeric = []
        for data in topic_data.values():
            timestamps = data['timestamps']
            if granularity == "Day":
                timestamps_numeric.extend([ts.timestamp() for ts in timestamps])
            elif granularity == "Week":
                timestamps_numeric.extend([ts.toordinal() for ts in timestamps])
            elif granularity == "Month":
                timestamps_numeric.extend([pd.Period(ts, freq='M').ordinal for ts in timestamps])
            elif granularity == "Year":
                timestamps_numeric.extend([ts.year for ts in timestamps])

        z_min, z_max = min(timestamps_numeric), max(timestamps_numeric)

        # Create a Plotly figure
        layout = go.Layout(scene=dict(xaxis=dict(range=[x_min, x_max]),
                                    yaxis=dict(range=[y_min, y_max]),
                                    zaxis=dict(range=[z_min, z_max]),
                                    xaxis_title='t-SNE Dimension 1',
                                    yaxis_title='t-SNE Dimension 2',
                                    zaxis_title='Timestamp',
                                    bgcolor='rgba(0,0,0,0)',
                                    aspectratio=dict(x=1, y=1, z=0.7)))


        fig = go.Figure(layout=layout)

        # Get the color palette
        if color_palette == 'Plotly':
            colors = px.colors.qualitative.Plotly
        elif color_palette == 'D3':
            colors = px.colors.qualitative.D3
        elif color_palette == 'Alphabet':
            colors = px.colors.qualitative.Alphabet
        else:
            colors = color_palette

        topics_to_show = sorted(topics_to_show)
        # Create a scatter plot for each topic
        for i, topic_id in enumerate(topics_to_show):
            if topic_id not in topic_data:
                print(f"Skipping Topic {topic_id} due to insufficient samples.")
                continue

            embeddings = topic_data[topic_id]['embeddings']
            timestamps = topic_data[topic_id]['timestamps']
            words = topic_data[topic_id]['words']

            # Cycle through colors if number of topics exceeds available colors
            topic_color = colors[i % len(colors)]

            # Get the topic name from the topic information
            topic_name = topic_info.loc[topic_info['Topic'] == topic_id, 'Name'].values[0]

            # Convert timestamps to numeric representation based on the selected granularity
            if granularity == "Day":
                timestamps_numeric = [ts.timestamp() for ts in timestamps]
            elif granularity == "Week":
                timestamps_numeric = [ts.toordinal() for ts in timestamps]
            elif granularity == "Month":
                timestamps_numeric = [pd.Period(ts, freq='M').ordinal for ts in timestamps]
            elif granularity == "Year":
                timestamps_numeric = [ts.year for ts in timestamps]

            scatter = go.Scatter3d(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                z=timestamps_numeric,
                mode='markers',
                name=topic_name,
                hovertext=[f'Topic: {topic_id}<br>Timestamp: {timestamp}<br>Words: {words[i]}' for i, timestamp in enumerate(timestamps)],
                hoverinfo='text',
                marker=dict(size=5, color=topic_color, opacity=0.7),
                visible='legendonly'  # Initially deactivated
            )
            fig.add_trace(scatter)

        fig.update_layout(
            title='Topic Evolution in Semantic Space (t-SNE)',
            legend_title='Topics',
            hovermode='closest',
            width=1200, height=800,
            legend=dict(
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            )
        )

        return fig



    def find_similar_topic_pairs_by_timestamp(self, similarity_threshold: float = 0.8) -> List[List[Tuple[int, int, str]]]:
        """
        Find topic pairs that become highly similar at specific timestamps.

        Parameters:
        - similarity_threshold: The cosine similarity threshold above which topics are considered highly similar.

        Returns:
        - A list of lists, where each sublist represents a group of topic-timestamp pairs that become highly similar.
        """
        topic_ids = self.final_df['Topic'].unique()
        num_topics = len(topic_ids)
        
        similar_topic_pairs_by_timestamp = []
        
        for i in range(num_topics):
            for j in range(i+1, num_topics):
                topic_i = topic_ids[i]
                topic_j = topic_ids[j]
                
                topic_i_data = self.final_df[self.final_df['Topic'] == topic_i]
                topic_j_data = self.final_df[self.final_df['Topic'] == topic_j]
                
                common_timestamps = set(topic_i_data['Timestamp']) & set(topic_j_data['Timestamp'])
                
                for timestamp in common_timestamps:
                    embedding_i = topic_i_data[topic_i_data['Timestamp'] == timestamp]['Embedding'].iloc[0]
                    embedding_j = topic_j_data[topic_j_data['Timestamp'] == timestamp]['Embedding'].iloc[0]
                    
                    similarity = cosine_similarity([embedding_i], [embedding_j])[0][0]
                    
                    if similarity >= similarity_threshold:
                        topic_pair = (topic_i, topic_j, timestamp)
                        
                        if not any(topic_pair in sublist for sublist in similar_topic_pairs_by_timestamp):
                            similar_topic_pairs_by_timestamp.append([topic_pair])
                        else:
                            for sublist in similar_topic_pairs_by_timestamp:
                                if any(pair[:2] == topic_pair[:2] for pair in sublist):
                                    sublist.append(topic_pair)
                                    break
        
        return similar_topic_pairs_by_timestamp
