#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
#

"""
Prompt name definition :

LANGUAGE_SYSTEM-OR-USER_PROMPT-DESCRIPTION

LANGUAGE: language used in the prompt.
SYSTEM-OR-USER: whether the prompt is to be send as the role of system or user.
PROMPT-DESCRIPTION: brief description of the prompt.

If positional parameters are used, write a description below the prompt.
"""

### BERTOPIC ###
FR_SYSTEM_SUMMARY_WORDS = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Générez un résumé en {num_words} mots maximum de ce texte."
    "Essayez de retenir les points les plus importants, "
    "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
    "les points principaux du texte sans avoir besoin de lire le texte en entier. "
    "Eviter les détails inutiles."
    )
# num_words: number of words the summary should contain

EN_SYSTEM_SUMMARY_WORDS = ("You are a highly qualified AI, trained in language understanding and synthesis. "
                           "Generate a summary of the following text in a maximum of {num_words} words."
                           "Try to capture the most important points, "
                           "providing a coherent and readable summary that could help a person understand "
                           "the main points of the text without needing to read the entire text. "
                           "Please avoid unnecessary details."
                           )
# num_words: number of words the summary should contain


FR_USER_SUMMARY_WORDS = FR_SYSTEM_SUMMARY_WORDS + (
						 " Texte :\n {text}"
						 )
# num_words: number of words the summary should contain
# text: text to be summarized

EN_USER_SUMMARY_WORDS = EN_SYSTEM_SUMMARY_WORDS + (
	" Text :\n {text}"
)
# num_words: number of words the summary should contain
# text: text to be summarized

FR_SYSTEM_SUMMARY_SENTENCES = FR_SYSTEM_SUMMARY_WORDS.replace("{num_words} mots", "{num_sentences} phrases")
EN_SYSTEM_SUMMARY_SENTENCES = EN_SYSTEM_SUMMARY_WORDS.replace("{num_words} words", "{num_sentences} sentences")
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

EN_USER_SUMMARY_MULTIPLE_DOCS = ("You are a highly qualified AI, trained in language understanding and synthesis. "
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

FR_USER_GENERATE_TOPIC_LABEL_TITLE = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Après utilisation d'un algorithme de topic modelling, un topic est représenté par les mots-clé suivants : \"\"\"{keywords}.\"\"\" "
    "Le topic contient plusieurs documents dont les titres sont les suivants :\n\"\"\"\n{title_list}\n\"\"\"\n"
    "À partir de ces informations sur le topic, écrivez un titre court de ce topic en 3 mots maximum. "
    )
# keywords: list of keywords describing the topic
# title_list: list of documents title belonging to the topic


FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES = ("Décrit en une courte expression le thème associé à l'ensemble des extraits "
                                          "suivants. Le thème doit être court et spécifique en 4 mots maximum. "
                                          "\n\"{title_list}\"")
# title_list: list of documents extracts belonging to the topic

EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES = (
    "Describe in a short sentence the topic associated with the following extracts. "
    "The topic description should be short and specific, no more than 4 words. "
    "\n\"{title_list}\"")
# title_list: list of documents extracts belonging to the topic

FRENCH_TOPIC_REPRESENTATION_PROMPT = ("J'ai un topic qui contient les documents suivants :\n"
									  "[DOCUMENTS]\n"
									  "Le topic est décrit par les mots-clés suivants : [KEYWORDS]\n"
									  "Sur la base des informations ci-dessus, extraire une courte étiquette de topic dans le format suivant :\n"
									  "Topic : <étiquette du sujet>")
# Passed directly to BERTopic's OpenAI wrapper, formatted similar to BERTopic's original prompt which can be found in its source code



### CHATBOT ###

"""
FR_USER_BASE_RAG = ("Répondez à la question en utilisant le contexte fourni. La réponse doit être {expected_answer_size}. "
					"Si le contexte ne fourni pas assez d'information pour répondre à la question, répondre : "
					"\"Le contexte fourni n'est pas suffisant pour répondre.\n"
					"---\n"
					"Contexte :\n"
					"\"\"\"\n"
					"{context}\n"
					"\"\"\"\n"
					"Question : {query}\n"
					)
"""
FR_USER_BASE_RAG = ("Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
                    "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
                    "À partir des documents fournis dans le contexte, répondez à la question. "
                    "La réponse doit être {expected_answer_size}, spécifique et percutante.\n"
                    '---\nContexte:\n'
                    '"""\n{context}\n"""\n---\n'
                    'Question : {query}')

# context : retrieved context
# expected_answer_size : size of the answer
# query : user query

FR_USER_MULTITURN_RAG = ("Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
                         "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
                         "À partir des documents fournis dans le contexte et de l'historique de la conversation, "
                         "répondez à la question finale. "
                         "La réponse doit être {expected_answer_size}, spécifique et percutante.\n"
                         '---\nContexte:\n'
                         '"""\n{context}\n"""\n---\n'
                         'Historique de conversation:\n'
                         '"""\n{history}\n"""\n---\n'
                         'Question finale : {query}')

# context : retrieved context
# expected_answer_size : size of the answer
# history : history of the conversation including queries and answers
# query : user query

FR_USER_MULTITURN_QUESTION_SPECIFICATION = ("Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
                                            "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
                                            "À partir de l'historique de conversation et de la dernière question de l'utilisateur, "
                                            "qui peut faire référence à l'historique de conversation, reformulez la dernière question "
                                            "de l'utilisateur pour qu'elle soit compréhensible sans l'historique de la conversation. "
                                            "Ne répondez PAS à la question. Reformulez la question si elle fait appel à des "
                                            "éléments de l'historique de la conversation. Sinon, renvoyez-la sans reformulation. "
                                            "---\nHistorique de conversation:\n"
                                            '"""\n{history}\n"""\n---\n'
                                            "Question finale de l'utilisateur: {query}\n"
                                            "Question finale de l'utilisateur reformulée :")

# history : history of the conversation including queries and answers
# query : user query

FR_SYSTEM_DODER_RAG = ("Vous êtes un assistant expert en réseau de transport de l'électricité "
                       "développé par l'entreprise RTE (Réseau de Transport de l'Électricité). "
                       "L'une des missions de RTE est de réaliser des études sur le réseau électrique en France. "
                       "Les informations concernant la manière de réaliser ces études sont contenues dans la "
                       "DODER (Documentation Opérationnelle du Domaine Etudes de Réseaux). "
                       "Votre rôle est de répondre aux questions en vous basant sur des extraits de la DODER "
                       "qui serviront de contexte. "
                       "Si le contexte ne contient pas d'éléments permettant de répondre à la question, "
                       "répondre \"Le contexte ne fourni pas assez d'information pour répondre à la question.\"")

FR_QUESTION_GENERATION = \
    """A partir des  éléments du contexte les plus pertinents, génère des questions associées.
---
Contexte:
\"\"\"
{context}
\"\"\"
"""

FR_USER_BASE_QUERY = ("Répondez à la question suivante. La réponse doit être spécifique et percutante.\n"
                      '---'
                      'Question : {query}')

FR_USER_BASE_MULTITURN_QUERY = ("À partir de l'historique de la conversation, répondez à la question finale. "
                                "La réponse doit être spécifique et percutante.\n"
                                '---'
                                'Historique de conversation:\n'
                                '"""\n{history}\n"""\n---\n'
                                'Question finale : {query}')

### LLAMA 3 ###

FR_SYSTEM_RAG_LLAMA3 = ("You are an helpful assistant developed by RTE (Réseau de Transport d'Électricité). "
                        "You help users answer questions based on internal RTE documents. "
                        "Based on the documents provided and the conversation history, answer the user query. "
                        "Don't start your answer with sentences like \"According to documents...\", "
                        "answer the query directly. Your answer must be in french.")

FR_USER_RAG_LLAMA3 = ("Documents:\n"
                      "```\n"
                      "{context}\n"
                      "```\n\n"
                      "Conversation history:\n"
                      "```\n"
                      "{history}\n"
                      "```\n\n"
                      "User query: {query}")

FR_SYSTEM_QUERY_CONTEXTUALIZATION_LLAMA3 = ("You are an helpful assistant that helps for query contextualization task. "
                                            "Based on the conversation history and user last query, "
                                            "which may refer to the conversation history, rephrase user last query "
                                            "to make it understandable without the conversation history. "
                                            "Do NOT answer the query. Rephrase the query if it draws on elements "
                                            "of the conversation history. Otherwise, send it back without rephrasing. "
                                            "Don't start your answer with sentences like \"Here is the rephrased query...\", "
                                            "answer with the rephrased query directly. "
                                            "The rephrased query must be in french.")

FR_USER_QUERY_CONTEXTUALIZATION_LLAMA3 = ("Conversation history:\n"
                                          "```\n"
                                          "{history}\n"
                                          "```\n\n"
                                          "User last query: {query}")
