#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from wattelse.bertopic.newsletter_features import generate_newsletter
from wattelse.bertopic.train import train_BERTopic
from wattelse.bertopic.utils import TIMESTAMP_COLUMN, clean_dataset, split_df_by_paragraphs, load_data
from wattelse.data_provider.curebot_provider import CurebotProvider
from wattelse.summary import GPTSummarizer

COLUMN_URL = "url"
MIN_TEXT_LENGTH = 150
EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"

def parse_data(files: List[UploadedFile]) -> pd.DataFrame:
    """Read a list of excel files and return a single dataframe containing the data"""
    dataframes = []

    with TemporaryDirectory() as tmpdir:

        for f in files:
            with NamedTemporaryFile(mode="wb") as tmp_file:
                # Copy data from BytesIO to the temporary file
                tmp_file.write(f.getvalue())
                if tmp_file is not None:
                    st.info(f"Analyse des articles de: {f.name}")
                    provider = CurebotProvider(Path(tmp_file.name))
                    articles = provider.get_articles()
                    #articles_path = Path(tmpdir) / (f.name + ".jsonl")
                    articles_path = Path("/tmp") / (f.name + ".jsonl")
                    provider.store_articles(articles, articles_path)
                    df = load_data(articles_path.absolute().as_posix()).sort_values(by=TIMESTAMP_COLUMN,
                                                                                 ascending=False).reset_index(
                        drop=True).reset_index()
                    dataframes.append(df)

        # Concat all dataframes
        df_concat = pd.concat(dataframes, ignore_index=True)
        df_concat = df_concat.drop_duplicates(subset=COLUMN_URL, keep="first")
        return df_concat


def split_data():
    st.session_state["df_split"] = (
        split_df_by_paragraphs(st.session_state["df"])
        .drop("index", axis=1)
        .sort_values(
            by=TIMESTAMP_COLUMN,
            ascending=False,
        )
        .reset_index(drop=True)
        .reset_index()
    )

    # Clean dataset using min_text_length
    st.session_state["df_split"] = clean_dataset(
        st.session_state["df"],
        MIN_TEXT_LENGTH,
    )


def train_model():
    st.session_state["topic_model"], st.session_state["topics"], _ = train_BERTopic(
        full_dataset=st.session_state["df_split"],
        indices=None,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        use_cache=False,
    )
    st.success("Création de topics... OK")


def create_newsletter():
    st.session_state["newsletter"] = generate_newsletter(
        topic_model=st.session_state["topic_model"],
        df=st.session_state["df"],
        topics=st.session_state["topics"],
        top_n_topics=st.session_state["newsletter_nb_topics"],
        top_n_docs=st.session_state["newsletter_nb_docs"],
        improve_topic_description=st.session_state["newsletter_improve_description"],
        summarizer_class=GPTSummarizer,
        summary_mode='topic',
    )
    st.success("Création de newsletter... OK")


def download_newsletter():
    pass


def main_page():
    """Main page rendering"""
    # title
    st.title('Wattelse topic')

    # uploader
    uploaded_files = st.file_uploader("Fichiers Excel (format Curebot)", accept_multiple_files=True)

    # check content
    if uploaded_files:
        st.session_state["df"] = parse_data(uploaded_files)
        st.success("Transformation des données... OK")

        # Affichage du contenu du fichier Excel
        with st.expander("Contenu des données", expanded=False):
            st.dataframe(st.session_state["df"])

        # split and clean data
        split_data()

        # Buttons for functionalities
        # Topic detection
        st.button("Détection des topics", on_click=train_model, key="topic_detection")

        # Newsletter creation
        if "topic_model" in st.session_state.keys():
            st.button("Génération de newsletter", on_click=create_newsletter)

            # Newsletter download
            st.button("Téléchargement newsletter", on_click=download_newsletter)


def options():
    with st.sidebar.form("parameters_sidebar"):
        st.title("Réglages")

        st.slider("Nombre de topics", min_value=1, max_value=10, value=5, key="newsletter_nb_topics")

        st.slider("Nombre d'articles par topics", min_value=1, max_value=10, value=5, key="newsletter_nb_docs")

        st.slider("Longueur des synthèses (# mots)", min_value=10, max_value=200, value=100, key="nb_words")

        parameters_sidebar_clicked = st.form_submit_button(
            "OK", type="primary",  # on_click=save_widget_state
        )


def main():
    options()
    main_page()


# Main
main()
