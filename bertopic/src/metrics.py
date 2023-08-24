from typing import Tuple, Dict

from loguru import logger
from statistics import geometric_mean
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from sklearn.cluster import KMeans

RANDOM_STATE = 666

TIME_WEIGHT = 0.05

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

    def degrees_of_diffusion(self, topic_i: int, tw: float = TIME_WEIGHT) -> Tuple[Dict, Dict]:
        # time periods
        periods = list(set(self.topics_over_time.Timestamp))
        periods.sort()
        n = len(periods)

        # time period
        j_dates = self.topics_over_time.query(f"Topic == {topic_i}")["Timestamp"]

        DoD_i = {}
        DF_i = {}
        for j_date in j_dates:
            # TODO/FIXME: NB. here we consider only the "main" topic - shall we consider the top-k topics describing each document?
            # Document frequency of topic i period j
            DF_ij = int(self.topics_over_time.query(f"Timestamp == '{j_date}'and Topic == {topic_i}")["Frequency"])

            # Total number of documents of the period j
            NN_j = self.topics_over_time.query(f"Timestamp == '{j_date}'")["Frequency"].sum()


            j = periods.index(j_date)
            DoD_ij = DF_ij / NN_j * (1 - tw * (n - j))
            DoD_i[j_date] = DoD_ij
            DF_i[j_date] = DF_ij

        return DoD_i, DF_i

    def TIM_map(self, tw: float = TIME_WEIGHT) -> pd.DataFrame:
        """Computes a Topic Issue Map"""
        topics = set(self.topic_model.topics_)
        map = []
        for i in topics:
            DoD_i, DF_i = self.degrees_of_diffusion(i, tw)
            # In TIM map, x-axis = average DF of topics; y-axis = DoD's growth's rate (geometric mean)
            DF_avg = np.mean(list(DF_i.values()))
            DoD_avg = geometric_mean(list(DoD_i.values()))
            map.append(
                {TIM_x: DF_avg, TIM_y: DoD_avg, "topic": i,
                 "topic_description": self.topic_model.get_topic_info(int(i)).Name.get(0)})

        return pd.DataFrame(map)

    def plot_TIM_map(self, tw: float = TIME_WEIGHT):
        """Plots the Topic Emergence Map"""
        TIM = self.TIM_map(tw)
        TIM = self.identify_signals(TIM, TIM_x, TIM_y)
        return self.scatterplot_with_annotations(TIM, TIM_x, TIM_y, "topic", "topic_description",
                                          "Topic Issue Map (TEM)", TIM_x,
                                          TIM_y)

    # def degrees_of_visibility(self, topic_i: int, tw: float = TIME_WEIGHT) -> Tuple[Dict, Dict]:
    #     # FIXME! Do not use it right now - no difference with degrees_of_diffusion
    #     # FIXME: Adapt it using keywords that characterizes topics?
    #     # time periods
    #     periods = list(set(self.topics_over_time.Timestamp))
    #     periods.sort()
    #     n = len(periods)
    #
    #     # time period
    #     j_dates = self.topics_over_time.query(f"Topic == {topic_i}")["Timestamp"]
    #
    #     DoV_i = {}
    #     TF_i = {}
    #     for j_date in j_dates:
    #         # Number of appearance of the topic i in period j
    #         TF_ij = int(self.topics_over_time.query(f"Timestamp == '{j_date}'and Topic == {topic_i}")["Frequency"])
    #
    #         ## Total number of documents where the topic i appears
    #         NN_j = int(self.topic_model.get_topic_info(topic_i)["Count"])
    #
    #         j = periods.index(j_date)
    #         DoV_ij = TF_ij / NN_j * (1 - tw * (n - j))
    #         DoV_i[j_date] = DoV_ij
    #         TF_i[j_date] = TF_ij
    #
    #     return DoV_i, TF_i
    #
    # def TEM_map(self, tw: float = TIME_WEIGHT) -> pd.DataFrame:
    #     """Computes a Topic Emergence Map"""
    #     topics = set(self.topic_model.topics_)
    #     map = []
    #     for i in topics:
    #         DoV_i, TF_i = self.degrees_of_visibility(i, tw)
    #         # In KEM map, x-axis = average TF of topics; y-axis = DoV's growth's rate (geometric mean)
    #         TF_avg = np.mean(list(TF_i.values()))
    #         DoV_avg = geometric_mean(list(DoV_i.values()))
    #         map.append(
    #             {TEM_x: TF_avg, TEM_y: DoV_avg, "topic": i,
    #              "topic_description": self.topic_model.get_topic_info(int(i)).Name.get(0)})
    #
    #     return pd.DataFrame(map)
    #
    # def plot_TEM_map(self, tw: float = TIME_WEIGHT) :
    #     """Plots the Topic Emergence Map"""
    #     TEM = self.TEM_map(tw)
    #     TEM = self.identify_signals(TEM, TEM_x, TEM_y)
    #     return self.scatterplot_with_annotations(TEM, TEM_x, TEM_y, "topic", "topic_description",
    #                                       "Topic Emergence Map (TEM)", TEM_x,
    #                                       TEM_y)

    @staticmethod
    def identify_signals(topic_map: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """Adds interpretable characteristics to topics by clustering them according to the two dimensions of the map"""

        # remove topic -1 as this is noise and may perturb the interpretation / scaling
        topic_map = topic_map[topic_map.topic != -1]

        def get_signal(row):
            """ensures correct human labeling, assuming cluster labels are put in a correct order"""
            x = row["x_clus"]
            y = row["y_clus"]
            if x:  # x True => x big
                return STRONG_SIGNAL if y else NOISY_SIGNAL  # y True => y big
            else:
                return WEAK_SIGNAL if y else LATENT_SIGNAL

        try:
            X = topic_map[[y_col]]
            kmeans_y = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init="auto").fit(X)
            topic_map["y_clus"] = kmeans_y.labels_.astype(bool)
            # Labels obtained by kmeans may not match the "right" class... we check with one example
            if topic_map.query("y_clus==0").iloc[0][y_col] > topic_map.query("y_clus==1").iloc[0][y_col]:
                logger.warning("Needed to change the clustering labels in dim y to obtain meaningful signal labels")
                topic_map["y_clus"] = ~topic_map["y_clus"]

            topic_map["x_clus"] = 9999
            for i in set(kmeans_y.labels_):
                Xi = topic_map.query(f"y_clus == {i}")[[x_col]]
                kmeans_xi = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init="auto").fit(Xi)
                df2 = topic_map.query(f"y_clus == {i}")
                df2["x_clus"] = kmeans_xi.labels_.astype(bool)
                if df2.query("x_clus==0").iloc[0][x_col] > df2.query("x_clus==1").iloc[0][x_col]:
                    logger.warning("Needed to change the clustering labels in dim x to obtain meaningful signal labels")
                    df2["x_clus"] = ~df2["x_clus"]
                topic_map.update(df2)

            topic_map["signal"] = topic_map.apply(get_signal, axis=1)
        except ValueError as e:
            logger.error(f"Cannot characterize signals: {e}")
            topic_map["signal"] = UKNOWN_SIGNAL
        return topic_map

    @staticmethod
    def scatterplot_with_annotations(df, x_col, y_col, label_col, hover_data, title, x_label, y_label):
        """Utility function to plat scatter plot"""

        # Create a scatter plot using Plotly
        fig = px.scatter(df, x=x_col, y=y_col, text=label_col, size_max=10, hover_data=hover_data, color=df.signal)

        # Add annotations
        fig.update_traces(textposition='top center')

        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=False
        )

        fig.add_vline(df[x_col].median())
        fig.add_hline(df[y_col].median())

        return fig
