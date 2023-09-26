import streamlit as st
from wattelse.gpt_rte import generation


# Get model name used for generation

model_name = generation.model_name

# Make the streamlit application

st.header('GPT du pauvre')
st.text(f"Using model: {model_name}")


col1, col2 = st.columns([3,1])
with col1:
    # Text area
    prompt = st.text_area("Prompt:", "", height=180, key="text")    

with col2:
    # Slider to select the number of tokens to generate
    max_new_tokens = st.select_slider("Generated tokens:", options=[1,2,4,8,16,32,64,128,256,512])


# Define the genration button and the function when clicked

def on_click():
    # When the button is clicked, generate a response and send it to the text_area
    response = generation.generate(prompt, max_new_tokens=max_new_tokens)
    st.session_state.text = response

st.button("Generate", on_click=on_click)

