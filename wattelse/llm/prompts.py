"""
Prompt name definition :

LANGUAGE_SYSTEM-OR-USER_PROMPT-DESCRIPTION

LANGUAGE: language used in the prompt.
SYSTEM-OR-USER: whether the prompt is to be send as the role of system or user.
PROMPT-DESCRIPTION: brief description of the prompt.

If positional parameters are used, write a description below the prompt.
"""

### BERTOPIC ###
FR_SYSTEM_SUMMARY_WORDS = ("Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
					 "J'aimerais que vous lisiez le texte suivant et que vous le résumiez en maximum {num_words} "
					 "mots. Essayez de retenir les points les plus importants, "
					 "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
					 "les points principaux du texte sans avoir besoin de lire le texte en entier. "
					 "Veuillez éviter les détails inutiles ou les points tangentiels."
					 )
# num_sentences: number of words the summary should contain

EN_SYSTEM_SUMMARY_WORDS = ("You are a highly qualified AI, trained in language understanding and synthesis. "
						"I would like you to read the following text and summarize it in a maximum of {num_words} "
						"words. Try to capture the most important points, "
						"providing a coherent and readable summary that could help a person understand "
						"the main points of the text without needing to read the entire text. "
						"Please avoid unnecessary details or tangential points."
						)
# num_sentences: number of words the summary should contain


FR_USER_SUMMARY_WORDS = FR_SYSTEM_SUMMARY_WORDS + (
						 " Texte :\n {text}"
						 )
# num_sentences: number of words the summary should contain
# text: text to be summarized

EN_USER_SUMMARY_WORDS = EN_SYSTEM_SUMMARY_WORDS + (
	" Text :\n {text}"
)
# num_sentences: number of words the summary should contain
# text: text to be summarized

FR_SYSTEM_SUMMARY_SENTENCES = FR_SYSTEM_SUMMARY_WORDS.replace("{num_words} mots", "{num_sentences} phrases")
EN_SYSTEM_SUMMARY_SENTENCES = EN_SYSTEM_SUMMARY_WORDS.replace("{num_words} words", "{num_sentences} sentences")
# num_sentences: number of sentences the summary should contain


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
											"qui peut faire référence à l'histoirique de conversation, reformulez la dernière question "
											"de l'utilisateur pour qu'elle soit comprégensible sans l'historique de la conversation. "
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
