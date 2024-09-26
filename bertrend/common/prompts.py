#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

###################### SUMMARY PROMPTS

FR_SYSTEM_SUMMARY_WORDS = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Générez un résumé en {num_words} mots maximum de ce texte."
    "Essayez de retenir les points les plus importants, "
    "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
    "les points principaux du texte sans avoir besoin de lire le texte en entier. "
    "Eviter les détails inutiles."
)
# num_words: number of words the summary should contain

EN_SYSTEM_SUMMARY_WORDS = (
    "You are a highly qualified AI, trained in language understanding and synthesis. "
    "Generate a summary of the following text in a maximum of {num_words} words."
    "Try to capture the most important points, "
    "providing a coherent and readable summary that could help a person understand "
    "the main points of the text without needing to read the entire text. "
    "Please avoid unnecessary details."
)
# num_words: number of words the summary should contain


FR_USER_SUMMARY_WORDS = FR_SYSTEM_SUMMARY_WORDS + (" Texte :\n {text}")
# num_words: number of words the summary should contain
# text: text to be summarized

EN_USER_SUMMARY_WORDS = EN_SYSTEM_SUMMARY_WORDS + (" Text :\n {text}")
# num_words: number of words the summary should contain
# text: text to be summarized

FR_SYSTEM_SUMMARY_SENTENCES = FR_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} mots", "{num_sentences} phrases"
)
EN_SYSTEM_SUMMARY_SENTENCES = EN_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} words", "{num_sentences} sentences"
)
# num_sentences: number of sentences the summary should contain

FR_USER_SUMMARY_MULTIPLE_DOCS = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Voici ci-dessous plusieurs articles de presse (Titre et Contenu). "
    "Tous les articles appartiennent au même thème représenté par les mots-clés suivants : {keywords}. "
    "Générez une synthèse en {nb_sentences} phrases maximum de ces articles qui doit être en lien avec le thème évoqué par les mots-clés. "
    "La synthèse doit permettre de donner une vision d'ensemble du thème sans lire les articles. "
    "Ne pas commencer par 'Les articles...' mais commencer directement la synthèse.\n"
    "Liste des articles :\n"
    "```{article_list}```\n"
    "Synthèse :"
)

EN_USER_SUMMARY_MULTIPLE_DOCS = (
    "You are a highly qualified AI, trained in language understanding and synthesis. "
    "Below are several press articles (Title and Content). "
    "All the articles belong to the same topic represented by the following keywords: {keywords}. "
    "Generate a summary of these articles, which must be related to the theme evoked by the keywords. "
    "The summary must not exceed {nb_sentences} sentences and should include the essential information from the articles.\n"
    "List of articles :\n"
    "```{article_list}```\n"
    "Summary :"
)
# keywords: list of keywords describing the topic
# list of articles and their title


###################### TOPIC PROMPTS

FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES = (
    "Décrit en une courte expression le thème associé à l'ensemble des extraits "
    "suivants. Le thème doit être court et spécifique en 4 mots maximum. "
    '\n"{title_list}"'
)

# title_list: list of documents extracts belonging to the topic

EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES = (
    "Describe in a short sentence the topic associated with the following extracts. "
    "The topic description should be short and specific, no more than 4 words. "
    '\n"{title_list}"'
)
# title_list: list of documents extracts belonging to the topic

FR_USER_GENERATE_TOPIC_LABEL_TITLE = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    'Après utilisation d\'un algorithme de topic modelling, un topic est représenté par les mots-clé suivants : """{keywords}.""" '
    'Le topic contient plusieurs documents dont les titres sont les suivants :\n"""\n{title_list}\n"""\n'
    "À partir de ces informations sur le topic, écrivez un titre court de ce topic en 3 mots maximum. "
)
# keywords: list of keywords describing the topic
# title_list: list of documents title belonging to the topic


FRENCH_TOPIC_REPRESENTATION_PROMPT = (
    "J'ai un topic qui contient les documents suivants :\n"
    "[DOCUMENTS]\n"
    "Le topic est décrit par les mots-clés suivants : [KEYWORDS]\n"
    "Sur la base des informations ci-dessus, extraire une courte étiquette de topic dans le format suivant :\n"
    "Topic : <étiquette du sujet>"
)
# Passed directly to BERTopic's OpenAI wrapper, formatted similar to BERTopic's original prompt which can be found in its source code
