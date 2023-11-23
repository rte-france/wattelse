"""
Prompt name definition :

LANGUAGE_SYSTEM-OR-USER_PROMPT-DESCRIPTION

LANGUAGE: language used in the prompt.
SYSTEM-OR-USER: whether the prompt is to be send as the role of system or user.
PROMPT-DESCRIPTION: brief description of the prompt.

If positional parameters are used, write a description below the prompt.
"""

### BERTOPIC ###

FR_USER_SUMMARY = ("Résume le texte délimité par des triples guillemets en quelques phrases. "
				   "Inclus uniquement les éléments importants et omets tous les détails :\n\"\"\"{text}\"\"\""
				   )
# text: text to be summarized

FR_SYSTEM_SUMMARY = ("Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
					 "J'aimerais que vous lisiez le texte suivant et que vous le résumiez en maximum {num_words} "
					 "mots. Essayez de retenir les points les plus importants, "
					 "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
					 "les points principaux de la discussion sans avoir besoin de lire le texte en entier. "
					 "Veuillez éviter les détails inutiles ou les points tangentiels."
					 )
# num_words: number of words the summary should contain

FR_USER_GENERATE_TOPIC_LABEL_TITLE = ("Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
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
FR_USER_BASE_RAG = ('Réponds à la question en résumant les éléments du contexte les plus pertinents. '
					'Ce résumé doit être {expected_answer_size}, spécifique et percutant.\n'
					'---\nContexte:\n'
					'"""\n{context}\n"""\n---\n'
					'Question : {query}')


# context : retrieved context
# expected_answer_size : size of the answer
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
