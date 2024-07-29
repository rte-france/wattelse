import os
import tempfile
import yaml
from pathlib import Path
from docxtpl import DocxTemplate

import streamlit as st

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

# Streamlit configs
st.set_page_config(layout="wide")
print(Path(__file__).parent / "RTE_logo.svg")
st.logo(str(Path(__file__).parent / "RTE_logo.svg"))

# Paths
EXAMPLE_D1_DIR_PATH = Path(__file__).parent / "example_D1"
RAG_QUERIES_FILE_PATH = Path(__file__).parent / "RAG_queries.yaml"
DCR_TEMPLATE_FILE_PATH = Path(__file__).parent / "Trame-DTR_streamlit.docx"

# RAG client config
RAG_CLIENT = RAGOrchestratorClient()
SESSION_NAME = "test_raccordement"
RAG_CLIENT.create_session(SESSION_NAME)

# Load RAG queries
with open(RAG_QUERIES_FILE_PATH, "r") as f:
    RAG_QUERIES_DICT = yaml.safe_load(f)


# Utils fonctions
def download_study(context: dict):
    """
    Fill in DCR template with the given context dict using Jinja2 style
    and show streamlit download button.
    """
    # Render template
    doc = DocxTemplate(DCR_TEMPLATE_FILE_PATH)
    doc.render(context)
    # Show download button
    with tempfile.TemporaryDirectory() as temp_dir:
        doc.save(temp_dir + "etude_test.docx")
        with open(temp_dir + "etude_test.docx", "rb") as f:
            st.download_button(
                "Télécharger Word", f, file_name="etude_test.docx", type="primary"
            )


def load_example_d1(file: Path):
    """Load example D1 information into `session_state`."""
    with open(file, "r") as f:
        d1_dict = yaml.safe_load(f)
        st.session_state.update(d1_dict)


def unit_RAG_query(
    label: str, template_key: str, prompt: str = None, study_name: str = None
):
    """
    Unit RAG element for information retrieval in the study.
    This fonction has to be called for every RAG element.
    """
    st.subheader(label)
    col1, col2 = st.columns(2)

    # RAG prompt
    with col1:
        st.text_area("Prompt", value=prompt, key="prompt_" + template_key)

    # RAG query execution button
    if st.button("Lancer", key="button_" + template_key, type="primary"):
        response = RAG_CLIENT.query_rag(
            SESSION_NAME,
            st.session_state["prompt_" + template_key],
            selected_files=[study_name],
        )
        st.session_state[template_key] = response["answer"]

    # RAG response
    with col2:
        st.text_area(
            "Réponse",
            value=st.session_state.get(template_key),
            key=template_key,
        )


# Streamlit front functions
def sidebar():
    with st.sidebar:
        st.title("POC synthèse DCR")
        st.write("---")
        if st.button("Générer Word"):
            download_study(st.session_state)


def D1_tab():
    st.title("Informations fiche D1")
    col1, col2 = st.columns(2)

    # D1 Form
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

    # Pre-filled D1
    with col2:
        st.subheader("Exemples de formulaires pré-remplis :")
        for file in EXAMPLE_D1_DIR_PATH.glob("*.yaml"):
            if not file.name.startswith("_"):
                st.button(file.name, on_click=load_example_d1, args=(file,))


def RAG_tab():
    st.title("RAG étude SER")

    # Upload SER study file
    uploaded_file = st.file_uploader("Étude SER")

    if uploaded_file is None:
        st.warning("Ajoutez l'étude SER correspondante pour continuer.")
        st.stop()
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            RAG_CLIENT.upload_files(SESSION_NAME, [temp_file_path])

    for key, value in RAG_QUERIES_DICT.items():
        unit_RAG_query(
            value["name"],
            key,
            prompt=value.get("prompt"),
            study_name=uploaded_file.name,
        )


def main():
    sidebar()
    tab1, tab2 = st.tabs(["Fiche D1", "RAG"])
    with tab1:
        D1_tab()
    with tab2:
        RAG_tab()


main()
