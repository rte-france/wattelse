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


# Groundedness (Pertinence contextuelle) : La question doit pouvoir être répondue à partir du contexte donné 
# pour eviter toute types d'hallucinations.

# QA generation prompt template
QA_GENERATION_PROMPT = """
Votre tâche est d'écrire une question factuelle et une réponse donnée un contexte.
Vous devez créer trois types de questions :

1. **Simple** : une question factuelle concise qui peut être répondue directement avec une information simple du contexte.
2. **Raisonnement** : une question qui nécessite un raisonnement ou une déduction à partir des éléments du contexte.
3. **Multi-contexte ** : une question qui intègre plusieurs éléments ou informations du contexte pour formuler la réponse.

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte. 
Cela signifie que votre question factuelle NE DOIT PAS mentionner quelque chose comme "selon le passage" ou "Dans le contexte".

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

# Not used needs rework #
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

# Not used needs rework #
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




                                                            # RAGAS Evaluation metrics


# Still in the Testing phase  

# CONTEXT_NDCG_PROMPT = """
# Sur la base d'une question, et d'une liste d'extraits du contexte, évaluez chaque extrait pour déterminer son utilité à répondre à la question. 
# Attribuez une note de pertinence de 0 à 3 à chaque extrait en fonction de son utilité :
# - 0 : Non pertinent
# - 1 : Peu pertinent
# - 2 : Pertinent
# - 3 : Très pertinent

# Les extraits plus utiles et apparaissant en tête doivent obtenir un score de pertinence plus élevé. 

# Calculez ensuite la moyenne des note de pertinence.

# Réponse:::
# Question : {question}
# Contexte : {retrieved_contexts}

# Évaluation : (votre raisonnement pour la pertinence, sous forme de texte)
# Score NDCG :
# Réponse:::
# """


FAITHFULNESS_EVAL_PROMPT = """
Évaluez si la réponse est fondée sur le contexte fourni, sans introduire d’informations non supportées.

Réponse:::
Évaluation : (Expliquez votre raisonnement en indiquant si la réponse est fidèle aux informations du contexte, en termes de pertinence et de suffisance. Identifiez explicitement les points de correspondance ou de divergence avec le contexte.)

Jugement : (Attribuez un jugement sous forme de nombre entre 1 et 5, selon les critères suivants:
- 1 : Très insuffisant – Réponse largement infidèle au contexte, avec des informations non supportées.
- 2 : Insuffisant – Éléments liés au contexte, mais présence d’informations non fondées.
- 3 : Passable – Informations pertinentes, mais quelques inexactitudes.
- 4 : Satisfaisant – Majoritairement fidèle, avec quelques détails manquants.
- 5 : Très satisfaisant – Entièrement fidèle et complète selon le contexte.

**Conseils pour l’évaluation :**
- Vérifiez si la réponse s'appuie exclusivement sur le contexte fourni sans introduire d’informations extérieures.
- Assurez-vous que la réponse reflète fidèlement les points principaux du contexte.

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Jugement :' dans votre réponse.

Voici la réponse à évaluer ainsi que le contexte fourni.

Réponse : {answer}
Contexte : {retrieved_contexts}
Réponse:::
"""

RETRIEVABILITY_EVAL_PROMPT = """
Évaluez si le contexte récupéré est pertinent et suffisant pour répondre à la question posée.

Réponse:::
Évaluation : (Indiquez si le contexte permet de répondre à la question et contient les informations nécessaires. Précisez si la proportion d'extraits non pertinents par rapport au total des extraits impacte la qualité de la réponse, et mentionnez tout manque d'exhaustivité.)

Jugement : (Attribuez un jugement sous forme de nombre entre 1 et 5, selon les critères suivants :
- 1 : Très insuffisant – Contexte principalement hors sujet, sans informations utiles.
- 2 : Insuffisant – Contexte partiellement pertinent, manque d'informations clés, avec de nombreux extraits non pertinents.
- 3 : Passable – Contexte globalement pertinent, mais dilué par plusieurs extraits non pertinents.
- 4 : Satisfaisant – Contexte majoritairement pertinent, avec seulement quelques extraits non pertinents qui n’affectent pas fortement la compréhension.
- 5 : Très satisfaisant – Contexte totalement pertinent et exhaustif, contenant toutes les informations nécessaires.

**Conseils pour l’évaluation :**
- Vérifiez si le contexte répond directement à la question et si les extraits sont pertinents pour la réponse.
- Évaluez si la présence d'extraits non pertinents nuit à la clarté et à la compréhension.

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Jugement :' dans votre réponse.

Voici la question ainsi que le contexte récupéré pour évaluation.

Question : {question}
Contexte : {retrieved_contexts}
Réponse:::
"""


## Needs rework, and the actual idea metric is challenging ##
CORRECTNESS_EVAL_PROMPT = """
Évaluez si la réponse est correcte, c’est-à-dire, si elle répond à la question posée en donnant les informations essentielles sans erreurs factuelles importantes.

Réponse:::
Évaluation : (Expliquez votre raisonnement de votre Jugement en indiquant si la réponse est correcte, en vous basant sur la question posée. Identifiez explicitement les points de correspondance ou de divergence avec la question supportant votre Jugement.)

Jugement : (Attribuez un jugement sous forme de nombre entre 1 et 5, selon les critères suivants:
- 1 : Très insuffisant – Largement incorrecte, avec des erreurs majeures.
- 2 : Insuffisant – Partiellement correcte, avec des erreurs significatives ou imprécisions.
- 3 : Passable – Répond globalement à la question, mais comporte plusieurs inexactitudes.
- 4 : Satisfaisant – Répond bien à la question, avec seulement quelques inexactitudes mineures.
- 5 : Très satisfaisant – Entièrement correcte, précise et parfaitement alignée avec la question.

**Conseils pour l’évaluation :**
- Vérifiez si la réponse aborde tous les points importants de la question sans omissions.
- Assurez-vous qu’il n’y a pas d’interprétations erronées ou d’informations hors sujet.
- Évitez de pénaliser la réponse pour des informations supplémentaires qui, bien que non nécessaires, n’introduisent pas d'erreurs ni de confusion.

Vous DEVEZ fournir des valeurs pour 'Évaluation :' et 'Jugement :' dans votre réponse.

Voici la question ainsi que la réponse pour évaluation.

Question : {question}
Réponse : {answer}
Réponse:::
"""