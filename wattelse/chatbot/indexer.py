import tempfile
from pathlib import Path

import streamlit as st
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from wattelse.chatbot import DATA_DIR
from wattelse.common.text_parsers.extract_text_from_MD import parse_md
from wattelse.common.text_parsers.extract_text_from_PDF import parse_pdf
from wattelse.common.text_parsers.extract_text_using_origami import parse_docx


def index_file(uploaded_file: UploadedFile):
    # NB: UploadedFile  can be used like a BytesIO
    extension = uploaded_file.name.split(".")[-1].lower()
    # create a temporary file using a context manager
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + "/" + uploaded_file.name, "wb") as file:
            # store temporary file
            file.write(uploaded_file.getbuffer())
            # extract data from file
            if extension == "pdf":
                st.session_state["data_files_from_parsing"].append(parse_pdf(
                    Path(file.name), DATA_DIR)
                )
            elif extension == "docx":
                st.session_state["data_files_from_parsing"].append(parse_docx(
                    Path(file.name), DATA_DIR
                ))
            elif extension == "md":
                st.session_state["data_files_from_parsing"].append(parse_md(
                    Path(file.name), DATA_DIR
                ))
            else:
                st.error("File type not supported!")


def index_files():
    uploaded_files = st.session_state["uploaded_files"]
    logger.debug(f"Uploading file: {uploaded_files}")
    for f in uploaded_files:
        index_file(f)
