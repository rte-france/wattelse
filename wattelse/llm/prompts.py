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
					 "J'aimerais que vous lisiez le texte suivant et que vous le résumiez en maximum {num_sentences} "
					 "phrases abstraites et concises. Essayez de retenir les points les plus importants, "
					 "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
					 "les points principaux de la discussion sans avoir besoin de lire le texte en entier. "
					 "Veuillez éviter les détails inutiles ou les points tangentiels."
					 )
# num_sentences: number of sentences the summary should contain

FR_USER_GENERATE_TOPIC_LABEL_TITLE = ("Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
									  "Après utilisation d'un algorithme de topic modelling, un topic est représenté par les mots-clé suivants : \"\"\"{keywords}.\"\"\" "
									  "Le topic contient plusieurs documents dont les titres sont les suivants :\n\"\"\"\n{title_list}\n\"\"\"\n"
									  "À partir de ces informations sur le topic, écrivez un titre court de ce topic en 3 mots maximum."
									  )
# keywords: list of keywords describing the topic
# title_list: list of documents title belonging to the topic


FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES = ("Décrit en une courte expression le thème associé à l'ensemble des extraits "
										  "suivants. Le thème doit être court et spécifique en 4 mots maximum. "
										  "\n\"{title_list}\"")



### CHATBOT ###
