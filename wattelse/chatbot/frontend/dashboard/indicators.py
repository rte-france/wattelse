#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.


import json

import numpy as np
import pandas as pd

from wattelse.chatbot.backend.rag_backend import RAGBackEnd

from wattelse.chatbot.frontend.dashboard.dashboard_utils import (
    METIERS_GROUP_NAME,
)


def _compute_file_indicators(group: str = None):
    """Compute the number of files and number of chunks in the RAG backend that belong to the given group.
    Args:
        group (str): Name of the given group (st.session_state["group"]). Default value is None, returns (Nan, Nan).

    Returns:
        _type_: _description_
    """
    if group and group != METIERS_GROUP_NAME:
        bak = RAGBackEnd(group_id=group, config="azure_20241216.toml")
        nb_files = len(bak.get_available_docs())
        nb_chunks = len(bak.document_collection.collection.get()["documents"])
    else:
        nb_files = np.NaN
        nb_chunks = np.NaN

    return nb_files, nb_chunks


def add_score_eval(msg_daily: pd.DataFrame) -> pd.DataFrame:
    msg_daily["score"] = (
        msg_daily["great"]
        + 2.0 / 3.0 * msg_daily["ok"]
        + msg_daily["missing_info"] / 3.0
    )
    return msg_daily


def exponential_smoothing(
    df: pd.DataFrame, x: pd.Series, alpha: pd.Series, alpha_0: float, ema_label: str
) -> pd.DataFrame:
    """given a series, smooth it with a moving exponential average.
    Args:
        df (pd.DataFrame): dataframe where the result is to be stored
        x (pd.Series): variable to be smoothed
        alpha (pd.Series): variable that governs the exponential decay
        alpha_0 (float) : scale of the exponential decay
        ema_label (str) : label of the column of df where is stored the result

    Returns:
        pd.DataFrame: returns df with the new column
    """

    df[ema_label] = 0.0
    df["accum_factor"] = 1.0
    exp_alpha = np.exp(-alpha / alpha_0)
    accum_prev = 0

    init = True
    for num, _ in df.iterrows():
        if init == True:
            init = False

            accum_prev = 1.0
            ema_prev = x.loc[num]
            df.loc[num, ema_label] = ema_prev
            df.loc[num, "accum_factor"] = 1.0
        else:
            accum_prev = 1.0 + accum_prev * exp_alpha.loc[num]
            ema_prev = x.loc[num] / accum_prev + (1.0 - 1 / accum_prev) * ema_prev
            df.loc[num, "accum_factor"] = accum_prev
            df.loc[num, ema_label] = ema_prev

    return df


def build_msg_df_over_time(
    filtered_df: pd.DataFrame, nb_reponse_lissage: str
) -> pd.DataFrame:
    """Given the filtered_df (from the sidebar), compute the nb of question each day, and their evaluations.

    Args:
        filtered_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: the dataframe whose index is days, and columns are
        [nb_questions, "great", "ok", "missing_info", "wrong", "non_evalue", "nb_feedback_long", "tx_feedback"]
    """
    message_daily = filtered_df.resample("D", on="answer_timestamp").agg(
        {"message": "count", "long_feedback_bool": "sum"}
    )

    # Resample data by day and count occurrences of each short_feedback value
    short_feedback_daily = (
        filtered_df.resample("D", on="answer_timestamp")["short_feedback"]
        .value_counts()
        .unstack()
    )

    message_daily = short_feedback_daily.join(message_daily)
    message_daily.fillna(value=0, inplace=True)
    message_daily.rename(
        columns={
            "message": "nb_questions",
            "": "non_evalue",
            "long_feedback_bool": "nb_feedback_long",
        },
        inplace=True,
    )
    for feedback in ["great", "ok", "missing_info", "wrong", "non_evalue"]:
        if feedback not in message_daily.columns:
            message_daily[feedback] = 0
        message_daily.fillna({feedback: 0}, inplace=True)

    message_daily["evalue"] = (
        message_daily["nb_questions"] - message_daily["non_evalue"]
    )
    message_daily["tx_feedback"] = (
        message_daily["evalue"] / message_daily["nb_questions"]
    )
    message_daily = add_score_eval(msg_daily=message_daily)
    alpha = message_daily["evalue"]
    alpha_0 = nb_reponse_lissage
    message_daily = exponential_smoothing(
        df=message_daily,
        x=message_daily["evalue"],
        alpha=alpha,
        alpha_0=alpha_0,
        ema_label="EMA_evalue",
    )
    message_daily = exponential_smoothing(
        df=message_daily,
        x=message_daily["score"],
        alpha=alpha,
        alpha_0=alpha_0,
        ema_label="EMA_score",
    )
    message_daily["smoothed_eval"] = (
        message_daily["EMA_score"] / message_daily["EMA_evalue"]
    )

    return message_daily


def build_users_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Given the log of each question, and evaluation, build a pivot table by user :
        - nb of question
        - nb of evaluation "great"
        - nb of evaluation "ok"
        - nb of evaluation "missing_info"
        - nb of evaluation "wrong"
        - nb of long feedback
        - proportion of answers evaluated

    Args:
        filtered_df (pd.DataFrame): raw log data

    Returns:
        pd.DataFrame: the pivot table described above.
    """
    users_feedback = pd.pivot_table(
        data=filtered_df[["username", "short_feedback", "answer_timestamp"]],
        index="username",
        values="answer_timestamp",
        columns="short_feedback",
        aggfunc="count",
    )
    data = filtered_df[
        ["username", "group_id", "conversation_id", "response", "long_feedback"]
    ]
    data.loc[:, "long_feedback_bool"] = (data["long_feedback"] != "").astype(int)

    users_df = data.groupby(
        by="username",
    ).agg(
        {
            "conversation_id": lambda x: len(x.unique()),
            "response": "count",
            "long_feedback_bool": "sum",
        }
    )
    users_df = users_df.join(users_feedback)
    users_df.fillna(value=0, inplace=True)

    for feedback in ["great", "ok", "missing_info", "wrong"]:
        if feedback not in users_df.columns:
            users_df[feedback] = 0
    users_df.rename(
        columns={
            "conversation_id": "nb_conversation",
            "response": "nb_questions",
            "long_feedback_bool": "nb_feedback_long",
            "": "non_evalue",
        },
        inplace=True,
    )
    users_df["evalue"] = users_df["nb_questions"] - users_df["non_evalue"]
    users_df.sort_values(by="evalue", ascending=False, inplace=True)

    users_df["tx_feedback"] = users_df["evalue"] / users_df["nb_questions"]
    users_df.reset_index(inplace=True)

    return users_df


def build_users_satisfaction_over_nb_eval(users_df: pd.DataFrame) -> pd.DataFrame:
    """Given the users_df, the dataframe of each users with their evaluations, this function gives
    an average evaluation for all users who has made X evaluations. This permits to visualize if evaluation
    changes with usage of the chatbot.
    The notation system is :
        - 0 point for "wrong" answer
        - 1 point for "missing" info
        - 2 points for "ok" answer
        - 3 points for "great" answer

    Args:
        users_df (pd.DataFrame): dataframe of each users with their evaluations

    Returns:
        pd.DataFrame: _description_
    """
    users_df["eval_mean"] = (
        users_df["great"] + 2.0 * users_df["ok"] / 3.0 + users_df["missing_info"] / 3.0
    ) / users_df["evalue"]
    users_df["eval_mean_std"] = users_df["eval_mean"]
    users_df["eval_std"] = np.sqrt(
        (
            users_df["great"] * (1 - users_df["eval_mean"]) ** 2
            + users_df["ok"] * (2.0 / 3.0 - users_df["eval_mean"]) ** 2
            + users_df["missing_info"] * (1.0 / 3.0 - users_df["eval_mean"]) ** 2
        )
        / users_df["evalue"]
    )
    users_satisfaction_over_nb_eval = pd.pivot_table(
        data=users_df,
        index="evalue",
        values=["eval_mean", "eval_mean_std", "eval_std"],
        aggfunc={"eval_mean": "mean", "eval_mean_std": "std", "eval_std": "mean"},
    )

    return users_satisfaction_over_nb_eval


def build_extracts_df(filtered_df: pd.DataFrame) -> pd.DataFrame:

    relevant_extracts = dict()
    columns_to_add = filtered_df.columns.tolist()
    columns_to_add.remove("relevant_extracts")

    for question_id, row in filtered_df.iterrows():
        json_data = json.loads(row["relevant_extracts"])
        for relevant_extract in json_data:
            relevant_extracts[f"{question_id}_{relevant_extract}"] = json_data[
                relevant_extract
            ].copy()
            del relevant_extracts[f"{question_id}_{relevant_extract}"]["metadata"]
            for metadata in json_data[relevant_extract]["metadata"]:
                relevant_extracts[f"{question_id}_{relevant_extract}"][metadata] = (
                    json_data[relevant_extract]["metadata"][metadata]
                )

            for col in columns_to_add:
                relevant_extracts[f"{question_id}_{relevant_extract}"][col] = row[col]
    relevant_extracts_df = pd.DataFrame.from_dict(relevant_extracts, orient="index")

    return relevant_extracts_df


def build_extracts_pivot(extracts_pivot: pd.DataFrame) -> pd.DataFrame:
    """Given the log of each question, and evaluation, build a pivot table by user :
        - nb of question
        - nb of evaluation "great"
        - nb of evaluation "ok"
        - nb of evaluation "missing_info"
        - nb of evaluation "wrong"
        - nb of long feedback
        - proportion of answers evaluated

    Args:
        filtered_df (pd.DataFrame): raw log data

    Returns:
        pd.DataFrame: the pivot table described above.
    """

    extracts_feedback = pd.pivot_table(
        data=extracts_pivot[["content", "short_feedback", "answer_timestamp"]],
        index="content",
        values="answer_timestamp",
        columns="short_feedback",
        aggfunc="count",
    )
    data = extracts_pivot[
        ["content", "group_id", "conversation_id", "response", "long_feedback"]
    ]
    data.loc[:, "long_feedback_bool"] = (data["long_feedback"] != "").astype(int)

    extracts_pivot = data.groupby(
        by="content",
    ).agg(
        {
            "conversation_id": lambda x: len(x.unique()),
            "response": "count",
            "long_feedback_bool": "sum",
        }
    )
    extracts_pivot = extracts_pivot.join(extracts_feedback)
    extracts_pivot.fillna(value=0, inplace=True)
    for feedback in ["great", "ok", "missing_info", "wrong"]:
        if feedback not in extracts_pivot.columns:
            extracts_pivot[feedback] = 0
    extracts_pivot["reponses correctes"] = (
        extracts_pivot["great"] + extracts_pivot["ok"]
    )
    extracts_pivot["reponses incorrectes"] = (
        extracts_pivot["missing_info"] + extracts_pivot["wrong"]
    )
    extracts_pivot["nb_evaluations"] = (
        extracts_pivot["reponses correctes"] + extracts_pivot["reponses incorrectes"]
    )

    extracts_pivot.sort_values(by="reponses incorrectes", ascending=False, inplace=True)
    extracts_pivot.rename(
        columns={
            "conversation_id": "nb_conversation",
            "response": "nb_questions",
            "long_feedback_bool": "nb_feedback_long",
            "": "non_evalue",
        },
        inplace=True,
    )

    extracts_pivot["tx_satisfaction"] = (
        extracts_pivot["reponses correctes"] / extracts_pivot["nb_evaluations"]
    )
    extracts_pivot.reset_index(inplace=True)

    return extracts_pivot
