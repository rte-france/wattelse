import streamlit as st
from loguru import logger
from abstractive_summarizer import AbstractiveSummarizer
from extractive_summarizer import ExtractiveSummarizer
from wattelse.summary.chatgpt_summarizer import GPTSummarizer

MODEL_LIST = [ 
    "origami_summarizer",
    "mrm8488/camembert2camembert_shared-finetuned-french-summarization",
    "csebuetnlp/mT5_multilingual_XLSum",
    "chat_gpt",
               ]

DEFAULT_TEXT = """RTE, sigle du Réseau de transport d'électricité, est le gestionnaire de réseau de transport français responsable du réseau public de transport d'électricité haute tension en France métropolitaine (la Corse n'est pas gérée par RTE). Sa mission fondamentale est d’assurer à tous ses clients l’accès à une alimentation électrique économique, sûre et propre. RTE connecte ses clients par une infrastructure adaptée et leur fournit tous les outils et services qui leur permettent d’en tirer parti pour répondre à leurs besoins. À cet effet, RTE exploite, maintient et développe le réseau à haute et très haute tension. Il est le garant du bon fonctionnement et de la sûreté du système électrique. RTE achemine l’électricité entre les fournisseurs d’électricité (français et européens) et les consommateurs, qu’ils soient distributeurs d’électricité ou industriels directement raccordés au réseau de transport. Plus de 105 000 km de lignes comprises entre 45 000 et 400 000 volts et 50 lignes transfrontalières connectent le réseau français à 33 pays européens, offrant ainsi des opportunités d’échanges d’électricité essentiels pour l’optimisation économique du système électrique.

RTE fait partie du Réseau européen des gestionnaires de réseau de transport d’électricité (ENTSO-E), organisation qui regroupe les gestionnaires de réseaux de transport à haute et très haute tension de 35 pays3. Elle découle de la fusion le 1er juillet 2009 de l'UCTE, de BALTSO, de NORDEL, d'ATSOI et d'UKTSOA.

Les réseaux de ces sociétés desservent, via les réseaux de distribution, une population d'environ 500 millions de personnes. Ils se décomposent en cinq grands systèmes synchrones : l'Europe continentale (à laquelle est rattachée la Turquie), les pays baltes, les pays nordiques, l'Irlande et la Grande-Bretagne.

Les lignes à basse et moyenne tension françaises ne sont pas du ressort de RTE. Elles sont essentiellement exploitées par Enedis (anciennement ERDF, filiale de distribution électrique d'EDF), mais aussi par d'autres entreprises locales de distribution (ELD) comme Électricité de Strasbourg, l'Usine d'électricité de Metz, ou encore Gascogne Énergies Services à Aire-sur-l’Adour dans les Landes."""

models = {}

@st.cache_resource
def get_summarizer(summary_model):
    """Instantiates models once, keep them in memory"""
    model = models.get(summary_model)
    if model is None:
        logger.info(f"Instantiating: {summary_model}")
        if summary_model == "chat_gpt":
            model = GPTSummarizer()
        elif summary_model == "origami_summarizer":
            model = ExtractiveSummarizer()
        else:
            model = AbstractiveSummarizer(summary_model)
        models[summary_model] = model
    return model

def app():

    st.title("Summarizer (abstractive & extractive")

    st.write(
        "    Le modèle chat_gpt n'est pas un modèle privé. \nMerci de l'utiliser EXCLUSIVEMENT sur des données publics."
        )

    summary_model = st.selectbox("summary model", MODEL_LIST, key="summary_model")
    summary_ratio = st.number_input("summary ratio", min_value=1, max_value=100, value=20, key="summary_ratio")/100

    text = st.text_area("Input text", DEFAULT_TEXT, height=180, key="text")
    summary = st.text_area("Summary", "", height=50, key="summary")

    def on_click():
        summarizer = get_summarizer(summary_model)
        summary = summarizer.generate_summary(text, summary_ratio)
        st.session_state.summary = summary
        logger.debug(summary)


    st.button("Summarize", on_click=on_click)



if __name__ == "__main__":
    app()