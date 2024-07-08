import os
import streamlit as st
import tempfile
import yaml

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

RAG = RAGOrchestratorClient()
SESSION_NAME = "test_raccordement"
RAG.create_session(SESSION_NAME)

with open("information.yaml", 'r') as f:
	INFO_LIST = yaml.safe_load(f)

st.write(INFO_LIST)
# Title
st.title("RAG étude SER")

# Sidebar
with st.sidebar:
	uploaded_file = st.file_uploader("Étude SER")

if uploaded_file is not None:
	with tempfile.TemporaryDirectory() as temp_dir:
		temp_file_path = os.path.join(temp_dir, uploaded_file.name)
		with open(temp_file_path, 'wb') as temp_file:
			temp_file.write(uploaded_file.getbuffer())
		print(temp_file_path)
		RAG.upload_files(SESSION_NAME, [temp_file_path])

# Function
def unit_RAG(label: str, prompt: str = None):
	with st.expander(label):
		st.text_input("prompt", value=prompt, key="prompt_"+label)
		if st.button("Lancer", key="button_"+label):
			answer = RAG.query_rag(SESSION_NAME, st.session_state["prompt_"+label])
			st.write(answer["answer"])


# Main connection solution
st.header("Solution de raccordement principale")

# Main information
st.subheader("Informations principales")
unit_RAG("Type de raccordement", prompt='Quel est le type de raccordement pour la solution envisagée ? Répond uniquement par "Antenne", "Piquage" ou "Coupure"."')
unit_RAG("Poste/Liaison")

# First table
st.subheader("Description de la solution de raccordement")
unit_RAG("Domaine de tension de raccordement de référence")
unit_RAG("Tension de raccordement ")
unit_RAG("Stratégie de raccordement retenue")
unit_RAG("Stratégies écratées")
unit_RAG("S3REnR")
unit_RAG("Nécessité d'un transfert de capacité dans le cadre du S3REnR")
unit_RAG("Consistance des travaux")
unit_RAG("Coût du Raccordement")
unit_RAG("Notas relatives au coût")
unit_RAG("Délai de raccordement")
unit_RAG("Notas relatives au délai")
unit_RAG("Installation soumise à des limitations temporaires")

# If storage
st.subheader("Si stockage")
unit_RAG("Offre de raccordement")
unit_RAG("Conclusion de l’étude de contraintes de transit")
unit_RAG("Travaux à la charge de RTE")
unit_RAG("Limitations")