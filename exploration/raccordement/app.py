import os
import tempfile
import yaml
from pathlib import Path

import streamlit as st

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient



PATH_TO_EXAMPLE_D1_DIR = Path(__file__).parent / "example_D1"

RAG_CLIENT = RAGOrchestratorClient()
SESSION_NAME = "test_raccordement"
RAG_CLIENT.create_session(SESSION_NAME)

with open("information.yaml", 'r') as f:
	INFO_DICT = yaml.safe_load(f)

# st.title("Automatisation note DCR")

tab1, tab2, tab3, tab4 = st.tabs(["Accueil", "Fiche D1", "RAG", "Note DCR"])

with tab1:
    st.write(
        "Cette application a pour objectif de démontrer la faisaibilité "
        "d'un processus d'automatisation de la rédaction des notes "
        "d'étude exploratoire DCR envoyées aux clients à partir d'une "
        "fiche D1 et de la note d'étude SER associée."
    )

with tab2:
    st.title("Informations fiche D1")
    with st.expander("Formulaire D1"):
        with st.form("D1"):
            # Study ID
            st.write("**Identifiant de l'étude**")
            st.text_input("Numéro", placeholder="AA-XXX", key="study_id")

            # Client information
            st.write("**Informations client**")
            st.selectbox("Civilité contact", ["Madame", "Monsieur"], key="client_civil_status")
            st.text_input("Nom contact", key="client_name")
            st.text_input("Adresse email contact", key="client_email_address")
            st.text_input("Nom société", key="client_company_name")

            # Study information
            st.write("**Informations sur l'étude**")
            st.date_input("Date de la demande", format="DD/MM/YYYY", key="request_date")
            st.text_input("Type de filière", key="connection_type")
            st.text_input("Puissance installée en injection", key="generation_capacity_installed")
            st.text_input("Puissance de raccordement en injection", key="generation_connection_power")
            st.text_input("Puissance de raccordement en soutirage", key="load_connection_power")
            st.date_input("Date de mise en service souhaitée", format="DD/MM/YYYY", key="desired_commissioning_date")

            # Geographical information
            st.write("**Informations géographiques du point de raccordement**")
            st.number_input("Numéro de département", min_value=1, max_value=100, step=1, key="department_number")
            st.text_input("Nom du département", key="departement_name")
            st.text_input("Nom de la commune", key="commune_name")

            # Form submit button
            st.form_submit_button("Valider")

        st.subheader("Exemples de formulaire pré-remplis :")

    for file in PATH_TO_EXAMPLE_D1_DIR.glob("*.yaml"):
        st.button(file.name)

with tab3:
    # Title
    st.title("RAG étude SER")

    # Sidebar
    uploaded_file = st.file_uploader("Étude SER")

    if uploaded_file is None:
        st.warning("Ajoutez l'étude SER correspondante pour continuer.")
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