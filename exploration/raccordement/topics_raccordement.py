#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List

import inspect
import pandas as pd
import streamlit as st
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from wattelse.bertopic.newsletter_features import generate_newsletter, md2html
from wattelse.bertopic.train import train_BERTopic
from wattelse.bertopic.utils import TIMESTAMP_COLUMN, clean_dataset, split_df_by_paragraphs, TEXT_COLUMN, TITLE_COLUMN, \
    URL_COLUMN
from wattelse.indexer.document_parser import parse_file
from wattelse.indexer.document_splitter import split_file
from wattelse.summary import GPTSummarizer

COLUMN_URL = "url"
MIN_TEXT_LENGTH = 150
EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"
TOP_N_WORDS = 5
#EMBEDDING_MODEL_NAME = "antoinelouis/biencoder-camembert-base-mmarcoFR"

css_style = Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css"

if "topic_detection_disabled" not in st.session_state:
    st.session_state.topic_detection_disabled = False
if "newsletter_disabled" not in st.session_state:
    st.session_state.newsletter_disabled = False
if "import_expanded" not in st.session_state:
    st.session_state.import_expanded = True
if "st.session_state.topic_expanded" not in st.session_state:
    st.session_state.topic_expanded = True


@st.cache_data
def parse_data_from_files(files: List[UploadedFile]) -> pd.DataFrame:
    """Read a list of excel files and return a single dataframe containing the data"""
    dataframes = []
    with TemporaryDirectory() as tmpdir:

        for f in files:

            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())

            if tmp_file is not None:
                with st.spinner(f"Split into documents: {f.name}"):
                    path = Path(tmpdir + "/" + f.name)
                    # Parse file
                    logger.debug(f"Parsing: {path}")
                    docs = parse_file(path)
                    # Split the file into smaller chunks as a list of Document
                    logger.debug(f"Chunking: {path}")
                    splits = split_file(path.suffix, docs)
                    logger.info(f"Number of chunks for file {f.name}: {len(splits)}")

                    # construct input data for bertopic
                    df = pd.DataFrame([s.page_content for s in splits], columns=[TEXT_COLUMN])
                    df[TIMESTAMP_COLUMN] = datetime.now()
                    df[TITLE_COLUMN] = ""
                    df[URL_COLUMN] = ""

                    dataframes.append(df)

        # Concat all dataframes
        df_concat = pd.concat(dataframes, ignore_index=True)
        #df_concat = df_concat.drop_duplicates(subset=COLUMN_URL, keep="first")
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
        st.session_state["df_split"],
        MIN_TEXT_LENGTH,
    )


def train_model():
    st.session_state["topic_model"], st.session_state["topics"], _ = train_BERTopic(
        full_dataset=st.session_state["df_split"],
        indices=None,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        use_cache=False,
        top_n_words=TOP_N_WORDS,
    )


def create_newsletter():
    with st.spinner("Création de la newsletter..."):
        st.session_state["newsletter"], _, _ = generate_newsletter(
            topic_model=st.session_state["topic_model"],
            df=st.session_state["df"],
            df_split=None,#st.session_state["df_split"],
            topics=st.session_state["topics"],
            top_n_topics=st.session_state["newsletter_nb_topics"],
            top_n_docs=st.session_state["newsletter_nb_docs"],
            improve_topic_description=True,
            summarizer_class=GPTSummarizer,
            summary_mode='topic',
            openai_model_name=st.session_state["openai_model_name"],
            nb_sentences=st.session_state["nb_sentences"]
        )


@st.experimental_dialog("Newsletter preview", width="large")
def preview_newsletter():
    content = md2html(st.session_state["final_newsletter"], css_style=css_style)
    st.html(content)


def import_data():
    with st.expander("**Import des données**", expanded=st.session_state.import_expanded):
        # uploader
       uploaded_files = st.file_uploader("Fichiers PDF, DOCX, XLSX, PPTX, HTML, MD, TXT", accept_multiple_files=True,
                                          help="Glisser/déposer dans cette zone les fichiers au format supporté")

    # check content
    if uploaded_files :
        st.session_state["df"] = parse_data_from_files(uploaded_files)

        # split and clean data
        if "df" in st.session_state:
            #split_data()
            # Clean dataset using min_text_length
            st.session_state["df"] = clean_dataset(
                st.session_state["df"],
                MIN_TEXT_LENGTH,
            )
            st.session_state['df_split'] = st.session_state["df"]
            logger.info(f"Size of dataset: {len(st.session_state['df_split'])}")


def display_data():
    if "df" in st.session_state:
        st.session_state.import_expanded = False
        # Affichage du contenu des données
        with st.expander("**Contenu des données**", expanded=False):
            st.dataframe(st.session_state["df"])
            st.download_button(
                "Save dataset",
                st.session_state["df_split"].to_csv(index=False).encode('utf-8'),
                "topic_dataset.csv",
                "text/csv",
                key='download-csv'
            )


def detect_topics():
    if "df_split" in st.session_state:
        st.session_state.import_expanded = False
        with st.expander("**Détection de topics**", expanded=st.session_state.topic_expanded):
            # Topic detection
            col1, col2 = st.columns(2)
            with col1:
                st.button("Détection des topics", on_click=train_model, key="topic_detection", type="primary",
                          disabled=st.session_state.topic_detection_disabled)
            with col2:
                if "topic_model" in st.session_state:
                    st.info(f"Nombre de topics: {len(st.session_state['topic_model'].get_topic_info()) - 1}")


def newsletter_creation():
    # Newsletter creation
    if "topic_model" in st.session_state.keys():
        st.session_state.topic_expanded = False
        with st.expander("**Création de la newsletter**", expanded=True):
            # st.session_state.topic_detection_disabled = True
            generation_button = st.button("Génération de newsletter", on_click=create_newsletter, type="primary",
                                          disabled=st.session_state.newsletter_disabled)

            # Edit manually newsletter
            if "newsletter" in st.session_state.keys():
                st.text_area(
                    "Contenu éditable de la newsletter (faire CTRL+ENTREE pour prendre en compte les modifications)",
                    value=st.session_state["newsletter"] if (
                                "final_newsletter" not in st.session_state or generation_button)
                    else st.session_state["final_newsletter"],
                    height=400, key="final_newsletter")

                if "final_newsletter" in st.session_state:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Preview", on_click=preview_newsletter)
                    with col2:
                        # Newsletter download
                        st.download_button("Téléchargement", file_name="newsletter.html",
                                           data=md2html(st.session_state["final_newsletter"], css_style=css_style),
                                           type="primary")


def main_page():
    """Main page rendering"""
    # title
    st.title('Wattelse - Topics des études')
    import_data()
    display_data()
    detect_topics()
    newsletter_creation()


def options():
    with st.sidebar:
        st.title("Réglages")

        st.slider("Nombre max de topics", min_value=1, max_value=15, value=10, key="newsletter_nb_topics")

        st.slider("Nombre max d'articles par topics", min_value=1, max_value=10, value=5, key="newsletter_nb_docs")

        st.slider("Longueur des synthèses (# phrases)", min_value=1, max_value=10, value=4, key="nb_sentences")

        st.selectbox("Moteur de résumé", ["wattelse-gpt35"], key="openai_model_name")
        #st.selectbox("OpenAI endpoint", ("gpt-3.5-turbo", "gpt-4o"), key="openai_model_name")



def main():
    options()
    main_page()


# Main
main()
