#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from statistics import geometric_mean
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic

RANDOM_STATE = 666

TIME_WEIGHT = 0.04

TEM_x = "Average topic frequency (TF)"
TEM_y = "Time weighted increasing rate of DoV"

TIM_x = "Average document frequency (DF)"
TIM_y = "Time weighted increasing rate of DoD"

WEAK_SIGNAL = "Weak signal"
LATENT_SIGNAL = "Latent signal"
STRONG_SIGNAL = "Strong signal"
NOISY_SIGNAL = "Well-known / Noise"
UKNOWN_SIGNAL = "?"


class TopicMetrics():
    """A set of metrics to describe the topics"""

    def __init__(self, topic_model: BERTopic, topics_over_time: pd.DataFrame):
        self.topic_model = topic_model
        self.topics_over_time = topics_over_time

    def degrees_of_visibility(self, topic_i: int, tw: float = TIME_WEIGHT) -> Tuple[Dict, Dict]:
        # time periods
        periods = list(set(self.topics_over_time.Timestamp))
        periods.sort()
        n = len(periods)

        # time period
        j_dates = self.topics_over_time.query(f"Topic == {topic_i}")["Timestamp"]

        DoV_i = {}
        DF_i = {}
        for j_date in j_dates:
            # TODO/FIXME: NB. here we consider only the "main" topic - shall we consider the top-k topics describing each document?
            # Document frequency of topic i period j
            DF_ij = int(self.topics_over_time.query(f"Timestamp == '{j_date}'and Topic == {topic_i}")["Frequency"])

            # Total number of documents of the period j
            NN_j = self.topics_over_time.query(f"Timestamp == '{j_date}'")["Frequency"].sum()


            j = periods.index(j_date)
            DoV_ij = DF_ij / NN_j * (1 - tw * (n - j))
            DoV_i[j_date] = DoV_ij
            DF_i[j_date] = DF_ij

        return DoV_i, DF_i

    def TEM_map(self, tw: float = TIME_WEIGHT) -> pd.DataFrame:
        """Computes a Topic Emergence Map"""
        topics = set(self.topic_model.topics_)
        map = []
        for i in topics:
            DoV_i, DF_i = self.degrees_of_visibility(i, tw)
            # In TIM map, x-axis = average DF of topics; y-axis = DoD's growth's rate (geometric mean)
            DF_avg = np.mean(list(DF_i.values()))
            DoV_avg = geometric_mean(list(DoV_i.values()))
            map.append(
                {TEM_x: DF_avg, TEM_y: DoV_avg, "topic": i,
                 "topic_description": self.topic_model.get_topic_info(int(i)).Name.get(0)})

        return pd.DataFrame(map)

    def plot_TEM_map(self, tw: float = TIME_WEIGHT):
        """Plots the Topic Emergence Map"""
        TEM = self.TEM_map(tw)
        TEM = self.identify_signals(TEM, TEM_x, TEM_y)
        return self.scatterplot_with_annotations(TEM, TEM_x, TEM_y, "topic", "topic_description",
                                          "Topic Emergence Map (TEM)", TEM_x, TEM_y)


    @staticmethod
    def identify_signals(topic_map: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """Adds interpretable characteristics to topics by clustering them according to the two dimensions of the map"""

        # remove topic -1 as this is noise and may perturb the interpretation / scaling
        topic_map = topic_map[topic_map.topic != -1]

        x_med = topic_map[x_col].median()
        y_med = topic_map[y_col].median()

        def assign_label(row):
            x_big = row[x_col] > x_med
            y_big = row[y_col] > y_med
            if x_big :
                return STRONG_SIGNAL if y_big else NOISY_SIGNAL
            else:
                return WEAK_SIGNAL if y_big else LATENT_SIGNAL

        topic_map["signal"] = topic_map.apply(assign_label, axis=1)

        return topic_map


    @staticmethod
    def scatterplot_with_annotations(df, x_col, y_col, label_col, hover_data, title, x_label, y_label, animation_frame=None, animation_group=None):
        """Utility function to plat scatter plot"""

        # Create a scatter plot using Plotly
        fig = px.scatter(df, x=x_col, y=y_col, text=label_col, size_max=10, hover_data=hover_data, color=df.signal,
                         animation_frame=animation_frame, animation_group=animation_group)

        # Add annotations
        fig.update_traces(textposition='top center')

        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=False
        )

        fig.add_vline(df[x_col].median(), line_dash="dash", line_width=1, line_color="grey")
        fig.add_hline(df[y_col].median(), line_dash="dash", line_width=1, line_color="grey")

        return fig
