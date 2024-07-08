import streamlit as st

st.title("Informations fiche D1")

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
	st.text_input("Nom du département", key="departement_number")
	st.text_input("Nom de la commune", key="commune_name")

	# Form submit button
	st.form_submit_button("Valider")