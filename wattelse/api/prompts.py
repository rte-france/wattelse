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

### CHATBOT ###
FR_USER_BASE_RAG = (
    "Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
    "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
    "À partir des documents fournis dans le contexte, répondez à la question. "
    "La réponse doit être {expected_answer_size}, spécifique et percutante.\n"
    "---\nContexte:\n"
    '"""\n{context}\n"""\n---\n'
    "Question : {query}"
)

# context : retrieved context
# expected_answer_size : size of the answer
# query : user query

FR_USER_MULTITURN_RAG = (
    "Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
    "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
    "À partir des documents fournis dans le contexte et de l'historique de la conversation, "
    "répondez à la question finale. "
    "La réponse doit être {expected_answer_size}, spécifique et percutante.\n"
    "---\nContexte:\n"
    '"""\n{context}\n"""\n---\n'
    "Historique de conversation:\n"
    '"""\n{history}\n"""\n---\n'
    "Question finale : {query}"
)

# context : retrieved context
# expected_answer_size : size of the answer
# history : history of the conversation including queries and answers
# query : user query

FR_USER_MULTITURN_QUESTION_SPECIFICATION = (
    "Vous êtes une IA experte qui aide les utilisateurs à répondre à des "
    "questions sur la base de documents provenant de l'entreprise RTE (Réseau de Transport de l'Électricité). "
    "À partir de l'historique de conversation et de la dernière question de l'utilisateur, "
    "qui peut faire référence à l'historique de conversation, reformulez la dernière question "
    "de l'utilisateur pour qu'elle soit compréhensible sans l'historique de la conversation. "
    "Ne répondez PAS à la question. Reformulez la question si elle fait appel à des "
    "éléments de l'historique de la conversation. Sinon, renvoyez-la sans reformulation. "
    "---\nHistorique de conversation:\n"
    '"""\n{history}\n"""\n---\n'
    "Question finale de l'utilisateur: {query}\n"
    "Question finale de l'utilisateur reformulée :"
)

# history : history of the conversation including queries and answers
# query : user query

FR_SYSTEM_DODER_RAG = (
    "Vous êtes un assistant expert en réseau de transport de l'électricité "
    "développé par l'entreprise RTE (Réseau de Transport de l'Électricité). "
    "L'une des missions de RTE est de réaliser des études sur le réseau électrique en France. "
    "Les informations concernant la manière de réaliser ces études sont contenues dans la "
    "DODER (Documentation Opérationnelle du Domaine Etudes de Réseaux). "
    "Votre rôle est de répondre aux questions en vous basant sur des extraits de la DODER "
    "qui serviront de contexte. "
    "Si le contexte ne contient pas d'éléments permettant de répondre à la question, "
    'répondre "Le contexte ne fourni pas assez d\'information pour répondre à la question."'
)

FR_QUESTION_GENERATION = """A partir des  éléments du contexte les plus pertinents, génère des questions associées.
---
Contexte:
\"\"\"
{context}
\"\"\"
"""

FR_USER_BASE_QUERY = (
    "Répondez à la question suivante. La réponse doit être spécifique et percutante.\n"
    "---"
    "Question : {query}"
)

FR_USER_BASE_MULTITURN_QUERY = (
    "À partir de l'historique de la conversation, répondez à la question finale. "
    "La réponse doit être spécifique et percutante.\n"
    "---"
    "Historique de conversation:\n"
    '"""\n{history}\n"""\n---\n'
    "Question finale : {query}"
)


### LLAMA 3 ###

"""
FR_SYSTEM_RAG_LLAMA3 : Guide l'assistant pour répondre aux questions des utilisateurs en utilisant les documents récupérés, en insistant 
sur la synthèse et en évitant les hallucinations.

FR_USER_RAG_LLAMA3 : Fournit à l'assistant la requête de l'utilisateur, l'historique de la conversation et les documents récupérés pour 
structurer la réponse.

FR_SYSTEM_QUERY_CONTEXTUALIZATION_LLAMA3 : Reformule les questions floues des utilisateurs pour les rendre compréhensibles sans avoir 
besoin de l'historique de conversation.

FR_USER_QUERY_CONTEXTUALIZATION_LLAMA3 : Structure l'historique de la conversation et la dernière question de l'utilisateur pour 
aider l'assistant à reformuler les questions ambiguës.
"""

FR_SYSTEM_RAG_LLAMA3 = (
    "You are a helpful assistant developed by RTE (Réseau de Transport d'Électricité). "
    "Your task is to answer user queries using information retrieved from internal RTE documents. "
    "Ensure that your responses are fully based on these documents. If no relevant passages are found, inform the user instead of providing an answer. "
    "Respond in clear, accurate French, without introductory phrases, and focus on synthesizing the information.\n"
    "Here are additional instructions the user wants you to follow:\n"
    "{group_system_prompt}"
)

FR_USER_RAG_LLAMA3 = (
    "Documents:\n"
    "```\n"
    "{context}\n"
    "```\n\n"
    "Conversation history:\n"
    "```\n"
    "{history}\n"
    "```\n\n"
    "User query: {query}\n\n"
    "Ensure that your response is accurate and directly related to the user's query. "
    "Generate a synthesized response that is fully grounded in the retrieved documents. "
)

FR_SYSTEM_QUERY_CONTEXTUALIZATION_LLAMA3 = (
    "You are a helpful assistant for query contextualization. "
    "Your task is to rephrase the user's last query based on the conversation history, making sure the query is understandable in isolation, without prior context. "
    "If the query references elements from the conversation, clearly rephrase it for standalone understanding; otherwise, use the original query. "
)

FR_USER_QUERY_CONTEXTUALIZATION_LLAMA3 = (
    "Conversation history:\n"
    "```\n"
    "{history}\n"
    "```\n\n"
    "User's last query: {query}\n\n"
    "Rephrase the user's query to make it understandable without relying on the conversation history."
)
