import streamlit as st

from wattelse.bertopic.app.app_utils import load_data_wrapper
from wattelse.bertopic.app.data_utils import data_overview, choose_data
from wattelse.bertopic.utils import DATA_DIR, TIMESTAMP_COLUMN

st.title("Browse data")

# Load selected DataFrame
choose_data(DATA_DIR, ["*.csv", "*.jsonl*"])

df = (
    load_data_wrapper(
        f"{st.session_state['data_folder']}/{st.session_state['data_name']}"
    )
    .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
    .reset_index(drop=True)
).reset_index()

st.dataframe(df, hide_index=True)

data_overview(df)
