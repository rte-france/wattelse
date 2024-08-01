import fitz
import streamlit as st
import tiktoken

from transformers import AutoTokenizer

from wattelse.api.openai.client_openai_api import OpenAI_Client

st.set_page_config(layout="wide")

LOCAL_API = OpenAI_Client(api_key="EMPTY", endpoint="http://localhost:8888/v1")
LOCAL_API.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_API.model_name)

OPENAI_API = OpenAI_Client(
    api_key="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"
)
OPENAI_API.model_name = "gpt-4o-mini"
OPENAI_TOKENIZER = enc = tiktoken.encoding_for_model(OPENAI_API.model_name)


# Utils functions
def pdf_to_text(uploaded_file):
    # Open the PDF file
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # Initialize an empty string to store the text
    full_text = ""

    # Iterate through each page
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)
        # Extract text from the page
        text = page.get_text()
        # Append the text to the full text string
        full_text += text
    return full_text.replace("\n", " ")


def handle_streaming_answer(stream):
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text is not None:
            yield text


# Streamlit app
st.title("Long context summarizer")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        text = pdf_to_text(uploaded_file)
        summarize_button = st.button("Summarize")

base_system_prompt = (
    "The user will provide you with text in triple quotes. "
    'Summarize this text in maximum 400 words with a prefix that says "Summary:". '
    "The summary must be in french."
)

st.text_area("System prompt", value=base_system_prompt, key="system_prompt")

col1, col2 = st.columns(2)

with col1:
    st.subheader(LOCAL_API.model_name)
    if uploaded_file:
        local_api_tokens_number = format(
            len(LOCAL_TOKENIZER.encode(text)), ","
        ).replace(",", " ")
        st.write(f"Number of tokens: {local_api_tokens_number}")
        if summarize_button:
            local_summary_stream = LOCAL_API.generate(
                f'"""{text}"""',
                system_prompt=st.session_state["system_prompt"],
                stream=True,
            )
            st.write_stream(handle_streaming_answer(local_summary_stream))

with col2:
    st.subheader(OPENAI_API.model_name)
    if uploaded_file:
        openai_api_tokens_number = format(
            len(OPENAI_TOKENIZER.encode(text)), ","
        ).replace(",", " ")
        st.write(f"Number of tokens: {openai_api_tokens_number}")
        if summarize_button:
            openai_summary_stream = OPENAI_API.generate(
                f'"""{text}"""',
                system_prompt=st.session_state["system_prompt"],
                stream=True,
            )
            st.write_stream(handle_streaming_answer(openai_summary_stream))