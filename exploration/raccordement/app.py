import os
import tempfile
import yaml
from pathlib import Path
from docxtpl import DocxTemplate

import streamlit as st

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

st.set_page_config(layout="wide")

PATH_TO_EXAMPLE_D1_DIR = Path(__file__).parent / "example_D1"

RAG_CLIENT = RAGOrchestratorClient()
SESSION_NAME = "test_raccordement"
RAG_CLIENT.create_session(SESSION_NAME)

st.logo("./RTE_logo.svg")

with open("RAG_information.yaml", "r") as f:
    INFO_DICT = yaml.safe_load(f)

def download_study(context: dict):
    doc = DocxTemplate("./Trame-DTR_streamlit.docx")
    doc.render(context)
    with tempfile.TemporaryDirectory() as temp_dir:
        doc.save(temp_dir + "etude_test.docx")
        with open(temp_dir + "etude_test.docx", "rb") as f:
            st.download_button("Télécharger Word", f, file_name="etude_test.docx")


# Sidebar
with st.sidebar:
    st.title("POC synthèse DCR")
    st.write("---")
    if st.button("Générer Word"):
        download_study(st.session_state)

# Main page
tab1, tab2 = st.tabs(["Fiche D1", "RAG"])

with tab1:
    st.title("Informations fiche D1")
    col1, col2 = st.columns(2)
    with col1:
        with st.form("D1"):
            st.text_input("ID étude", placeholder="AA-XXX", key="D1_id_etude")
            st.selectbox(
                "Civilité contact", ["Madame", "Monsieur"], key="D1_civilité_contact"
            )
            st.text_input(
                "Prénom Nom contact",
                placeholder="Prénom Nom",
                key="D1_prenom_nom_contact",
            )
            st.text_input(
                "Adresse email contact", placeholder="xxx@yyy", key="D1_email_contact"
            )
            st.text_input("Nom société", placeholder="xxx", key="D1_nom_societe")

            st.text_input(
                "Date de la demande", placeholder="DD/MM/YYYY", key="D1_date_demande"
            )
            st.text_input("Type de filière", placeholder="xxx", key="D1_type_filiere")
            st.number_input(
                "Puissance de raccordement en injection (MW)",
                min_value=1,
                max_value=100,
                step=1,
                value=10,
                key="D1_puissance_injection",
            )
            st.number_input(
                "Puissance de raccordement en soutirage (MW)",
                min_value=1,
                max_value=100,
                step=1,
                value=10,
                key="D1_puissance_soutirage",
            )
            st.text_input(
                "Date de mise en service souhaitée",
                placeholder="DD/MM/YYYY",
                key="D1_mise_service",
            )
            st.number_input(
                "Numéro de département",
                min_value=1,
                max_value=100,
                step=1,
                key="D1_numero_departement_raccordement",
            )
            st.text_input(
                "Nom de la commune", placeholder="xxx", key="D1_commune_raccordement"
            )

            # Form submit button
            st.form_submit_button("Valider")

    def load_example_d1(file: Path):
        with open(file, "r") as f:
            d1_dict = yaml.safe_load(f)
            st.session_state.update(d1_dict)

    with col2:
        st.subheader("Exemples de formulaire pré-remplis :")
        for file in PATH_TO_EXAMPLE_D1_DIR.glob("*.yaml"):
            if not file.name.startswith("_"):
                st.button(file.name, on_click=load_example_d1, args=(file,))
            

with tab2:
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
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            print(temp_file_path)
            RAG_CLIENT.upload_files(SESSION_NAME, [temp_file_path])

    # Function
    def unit_RAG(label: str, prompt: str = None):
        st.subheader(label)
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Prompt", value=prompt, key="prompt_" + label)
        if st.button("Lancer", key="button_" + label):
            response = RAG_CLIENT.query_rag(
                SESSION_NAME,
                st.session_state["prompt_" + label],
                selected_files=[uploaded_file.name],
            )
            # st.write("Réponse :")
            # st.write("**"+response["answer"]+"**")
            st.session_state["answer_" + label] = response["answer"]
        with col2:
            st.text_area(
                "Réponse",
                value=st.session_state.get("answer_" + label),
                key="answer_" + label,
            )

    # Main connection solution

    for key, value in INFO_DICT.items():
        unit_RAG(value["name"], prompt=value.get("prompt"))
