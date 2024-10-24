#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

EVAL_LLM_PROMPT = (
    "You are an evaluator. You will be provided with a a query, "
    "the groundtruth answer and a candidate response. You must "
    "evaluate the candidate response based on the groundtruth "
    "answer and the query. You must provide one of the following "
    "scores:\n"
    "- 0: the candidate response is incorrect and contains wrong information\n"
    "- 1: the candidate response is not incorrect but miss important parts "
    "of the groundtruth answer\n"
    "- 2: the candidate response is correct but miss some details that "
    "do not impact the veracity of the information\n"
    "- 3: the candidate response is correct and provides all the "
    "information from the groundtruth answer\n"
    'You must answer with the score only, using the format "Score: {{0,1,2,3}}"\n\n'
    "Query: {query}\n"
    "Groundtruth answer: {answer}\n"
    "Candidate response: {candidate}\n"
)

FAITHFULNESS_EVAL_PROMPT = '''
Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict 
as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
'''


# Groundedness (Pertinence contextuelle) : La question doit pouvoir être répondue à partir du contexte donné. 
# Si elle nécessite des informations extérieures, elle n'est pas pertinente.

# Realism (Réalisme de la question utilisateur) : La question doit ressembler à une question qu'un utilisateur poserait en se basant sur le document. 
# Elle doit être réaliste et pertinente pour l'utilisateur.

# Stand-alone (Indépendance) : La question doit être compréhensible sans dépendre du contexte spécifique. 
# Elle doit avoir du sens pour quelqu'un ayant des connaissances générales du domaine.

# Pertinence contextuelle : Évalue si la question peut être répondue avec le contexte donné.
# Réalisme : Vérifie si la question ressemble à une question qu’un utilisateur poserait en fonction du document.
# Indépendance : Mesure si la question a du sens sans dépendre du contexte.


# QA generation prompt template
QA_GENERATION_PROMPT = """
Votre tâche est d'écrire une question factuelle et une réponse donnée un contexte.
Vous devez créer trois types de questions :

1. **Simple** : une question factuelle qui peut être répondue directement avec une information simple du contexte.
2. **Raisonnement** : une question qui nécessite un raisonnement ou une déduction à partir des éléments du contexte.
3. **Multi-contexte ** : une question qui intègre plusieurs éléments ou informations du contexte pour formuler la réponse.

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte. 
Cela signifie que votre question factuelle NE DOIT PAS mentionner quelque chose comme "selon le passage" ou "le contexte".

Fournissez votre réponse comme suit :

Sortie:::
Question simple : (votre question simple)
Réponse simple : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question simple)

Question de raisonnement : (votre question de raisonnement)
Réponse de raisonnement : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question de raisonnement)

Question multi-contexte : (votre question multi-contexte)
Réponse multi-contexte : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question multi-contexte)

Voici maintenant le contexte.

Contexte : {context}\n
Sortie:::
"""


QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT = """
Avec ce contexte, est-ce que je peux répondre à cette question ?

Réponse:::
Évaluation : (votre raisonnement pour la note, sous forme de texte)
Note totale : (votre note, sous forme de nombre entre 1 et 5)

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

Voici maintenant la question et le contexte.

Question : {question}\n
Contexte : {context}\n
Réponse:::
"""

# QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT = """
# Vous allez recevoir un contexte et une question.
# Votre tâche est d'évaluer si la question donnée peut être répondue sans ambiguïté en se basant uniquement sur les informations explicites présentes dans le contexte.
# Donnez votre réponse sur une échelle de 1 à 5, où 1 signifie que la question n'est pas du tout répondable compte tenu du contexte ou la question n'est pas pertinente par rapport au contexte, et 5 signifie que les informations nécessaires sont présentes de manière explicite

# Fournissez votre réponse comme suit :

# Réponse:::
# Évaluation : (votre raisonnement pour la note, sous forme de texte)
# Note totale : (votre note, sous forme de nombre entre 1 et 5)

# Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

# Voici maintenant la question et le contexte.

# Question : {question}\n
# Contexte : {context}\n
# Réponse:::"""


QUESTION_REALISM_CRITIQUE_PROMPT = """
Avec ce contexte, est-ce que cette question pourrait-elle être pertinente par un utilisateur du document ?

Réponse:::
Évaluation : (votre raisonnement pour la note, sous forme de texte)
Note totale : (votre note, sous forme de nombre entre 1 et 5)

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

Voici maintenant la question et le contexte.

Question : {question}\n
Contexte : {context}\n
Réponse:::
"""

# QUESTION_REALISM_CRITIQUE_PROMPT = """

# Vous allez recevoir une question.
# Votre tâche est de fournir une 'note totale' représentant à quel point cette question ressemble à une question qu'un utilisateur poserait en se basant sur le contenu du document.
# Donnez votre réponse sur une échelle de 1 à 5, où 1 signifie que la question ne pourrait jamais être posée par un utilisateur, et 5 signifie que la question est très réaliste et pertinente pour un utilisateur du document.

# Fournissez votre réponse comme suit :

# Réponse:::
# Évaluation : (votre raisonnement pour la note, sous forme de texte)
# Note totale : (votre note, sous forme de nombre entre 1 et 5)

# Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

# Voici maintenant la question.

# Question : {question}\n
# Réponse:::
# """

QUESTION_STANDALONE_CRITIQUE_PROMPT = """

Est-ce que cette question pourrait-elle être pertinente par un utilisateur du document ?

Réponse:::
Évaluation : (votre raisonnement pour la note, sous forme de texte)
Note totale : (votre note, sous forme de nombre entre 1 et 5)

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

Voici maintenant la question et le contexte.

Question : {question}\n
Réponse:::
"""


# QUESTION_STANDALONE_CRITIQUE_PROMPT = """
# Vous allez recevoir une question.
# Votre tâche est de fournir une 'note totale' représentant à quel point cette question est indépendante du contexte.
# Donnez votre réponse sur une échelle de 1 à 5, où 1 signifie que la question dépend d'informations supplémentaires pour être comprise, et 5 signifie que la question a un sens par elle-même.
# Par exemple, si la question fait référence à un cadre particulier, comme 'dans le contexte' ou 'dans le document', la note doit être 1.
# Les questions peuvent contenir des noms techniques obscurs ou des acronymes et recevoir tout de même une note de 5 : il doit simplement être clair aux personnes concernées ayant accès à la documentation.

# Par exemple, "Quel est le nom du point de contrôle à partir duquel le modèle ViT est importé ?" devrait recevoir une note de 1, car il y a une mention implicite d'un contexte, donc la question n'est pas indépendante du contexte.

# Fournissez votre réponse comme suit :

# Réponse:::
# Évaluation : (votre raisonnement pour la note, sous forme de texte)
# Note totale : (votre note, sous forme de nombre entre 1 et 5)

# Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

# Voici maintenant la question.

# Question : {question}\n
# Réponse:::"""




                                                            # RAGAS Evaluation metrics



CONTEXT_PRECISION_PROMPT = """
Indiquez si chaque extraits recuperé du contexte est pertinent pour la question, et notez la précision (le ratio d'extraits pertinents sur le total d'extraits).

Fournissez votre réponse comme suit :

Réponse:::
Évaluation : (raisonnement pour la pertinence des extraits et du contexte global)
Précision : (valeur entre 0 et 1)
Nombre total de extraits : {total_chunks}
Extraits pertinents : (nombre de fragments pertinents)
Extraits non-pertinents : (nombre de extraits non-pertinents)

Voici la question et le contexte.

Question : {question}
Contexte : {retrieved_contexts}
Réponse:::
"""



CONTEXT_GROUNDEDNESS_PROMPT = """
Avec ce contexte, est-ce que je peux répondre à cette question ?

Réponse:::
Évaluation : (votre raisonnement pour la note, sous forme de texte)
Note totale : (votre note, sous forme de nombre entre 1 et 5)

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Note totale :' dans votre réponse.

Voici maintenant la question et le contexte.

Question : {question}\n
Contexte : {retrieved_contexts}\n
Réponse:::
"""

# CONTEXT_PRECISION_PROMPT = """
# Votre tâche est d'analyser chaque extrait du contexte pour déterminer s'il fournit une information pertinente pour répondre à la question. Si ce n'est pas le cas, l'extrait n'est pas pertinent. Ensuite, calculez la précision comme suit : Précision = (Nombre d'extraits pertinents) / {total_chunks}.

# Fournissez votre réponse de la manière suivante :

# Évaluation : (Explication détaillée sur la pertinence de chaque extrait)
# Précision : (Valeur entre 0 et 1, par exemple : 8 pertinents / 10 extraits = 0,8)

# Voici la question et le contexte.

# Question : {question}
# Contexte : {retrieved_contexts}
# """



CONTEXT_PRECISION_WITH_REFERENCE_CRITIQUE_PROMPT = """
Vous allez recevoir une question, une réponse de référence, et un ensemble de contextes récupérés.
Votre tâche est d'évaluer dans quelle mesure chaque contexte récupéré est pertinent par rapport à la réponse de référence et contribue à répondre correctement à la question.
Pour chaque contexte, donnez une note sur une échelle de 1 à 5, où 1 signifie que le contexte n'est pas du tout pertinent par rapport à la réponse de référence, et 5 signifie que le contexte est extrêmement pertinent et aide directement à répondre à la question en s'appuyant sur la réponse de référence.

Fournissez votre réponse comme suit :

Réponse:::
Contexte 1 : (votre raisonnement pour la pertinence du premier contexte par rapport à la réponse de référence)
Note : (votre note pour le premier contexte, sous forme de nombre entre 1 et 5)

Contexte 2 : (votre raisonnement pour la pertinence du deuxième contexte par rapport à la réponse de référence)
Note : (votre note pour le deuxième contexte, sous forme de nombre entre 1 et 5)

Continuez ainsi pour chaque contexte récupéré.

Voici la question, la réponse de référence et les contextes récupérés.

Question : {question}
Réponse de référence : {reference}
Contextes récupérés : {retrieved_contexts}
Réponse:::
"""



CONTEXT_RECALL_WITH_REFERENCE_CRITIQUE_PROMPT = """
Vous allez recevoir une question, une réponse de référence, et un ensemble de contextes récupérés.
Votre tâche est d'évaluer dans quelle mesure les contextes récupérés couvrent toutes les informations importantes présentes dans la réponse de référence. 
Pour ce faire, identifiez chaque affirmation clé dans la réponse de référence et évaluez si elle est attribuable à l'un des contextes récupérés.
Donnez une note sur une échelle de 0 à 1 pour chaque contexte récupéré, où 1 signifie que toutes les affirmations de la réponse de référence sont présentes dans les contextes récupérés, et 0 signifie qu'aucune affirmation clé n'est couverte.

Fournissez votre réponse comme suit :

Réponse:::
Affirmation 1 (extraite de la réponse de référence) : (votre raisonnement pour déterminer si cette affirmation est couverte ou non par les contextes récupérés)
Note : (votre note, sous forme de nombre entre 0 et 1)

Affirmation 2 (extraite de la réponse de référence) : (votre raisonnement pour déterminer si cette affirmation est couverte ou non par les contextes récupérés)
Note : (votre note, sous forme de nombre entre 0 et 1)

Continuez ainsi pour chaque affirmation clé de la réponse de référence.

Voici la question, la réponse de référence et les contextes récupérés.

Question : {question}
Réponse de référence : {reference}
Contextes récupérés : {retrieved_contexts}
Réponse:::
"""



CONTEXT_RECALL_WITHOUT_LLM_CRITIQUE_PROMPT = """
Vous allez recevoir un ensemble de contextes récupérés et un ensemble de contextes de référence.
Votre tâche est d'évaluer dans quelle mesure les contextes récupérés couvrent les informations présentes dans les contextes de référence.
Pour chaque contexte de référence, évaluez s'il est couvert par un ou plusieurs des contextes récupérés et donnez une note sur une échelle de 0 à 1, où 1 signifie que le contexte de référence est parfaitement couvert, et 0 signifie qu'il n'est pas du tout couvert.

Fournissez votre réponse comme suit :

Réponse:::
Contexte de référence 1 : (votre raisonnement pour déterminer si ce contexte est couvert ou non par les contextes récupérés)
Note : (votre note, sous forme de nombre entre 0 et 1)

Contexte de référence 2 : (votre raisonnement pour déterminer si ce contexte est couvert ou non par les contextes récupérés)
Note : (votre note, sous forme de nombre entre 0 et 1)

Continuez ainsi pour chaque contexte de référence.

Voici les contextes récupérés et les contextes de référence.

Contextes récupérés : {retrieved_contexts}
Contextes de référence : {reference_contexts}
Réponse:::
"""



# Context Recall: How well the retrieved information covers the relevant context.
# Context Precision: The accuracy of the retrieved chunks in matching the query.
# Context Relevancy: Balances precision and recall to measure relevancy.
# Context Entity Recall: Evaluates the retrieval of specific entities mentioned in the query.