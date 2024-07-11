import os
import streamlit as st
import tempfile
import yaml

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

RAG_CLIENT = RAGOrchestratorClient()
SESSION_NAME = "test_raccordement"
RAG_CLIENT.create_session(SESSION_NAME)

with open("information.yaml", 'r') as f:
	INFO_DICT = yaml.safe_load(f)

# Title
st.title("RAG étude SER")

# Sidebar
with st.sidebar:
	uploaded_file = st.file_uploader("Étude SER")

if uploaded_file is None:
	st.warning("Ajoutez l'étude SER correspondante dans la barre de gauche.")
	st.stop()
else:
	with tempfile.TemporaryDirectory() as temp_dir:
		temp_file_path = os.path.join(temp_dir, uploaded_file.name)
		with open(temp_file_path, 'wb') as temp_file:
			temp_file.write(uploaded_file.getbuffer())
		print(temp_file_path)
		RAG_CLIENT.upload_files(SESSION_NAME, [temp_file_path])

# Function
def unit_RAG(label: str, prompt: str = None):
	with st.expander(label):
		st.text_area("Prompt", value=prompt, key="prompt_"+label)
		if st.button("Lancer", key="button_"+label):
			response = RAG_CLIENT.query_rag(SESSION_NAME, st.session_state["prompt_"+label], selected_files=[uploaded_file.name])
			st.write("Réponse :")
			st.write("**"+response["answer"]+"**")
			st.session_state["answer_"+label] = response["answer"]


# Main connection solution
st.header("Solution de raccordement principale")

for key, value in INFO_DICT.items():
	st.subheader(key)
	for info in value:
		unit_RAG(info["name"], prompt=info.get("prompt"))