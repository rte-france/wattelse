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
Votre tâche est d'écrire une question factuelle et sa réponse en fonction d'un contexte donné.
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
Evaluate whether the response is based on the provided context, without introducing unsupported information.

Response:::
Evaluation: (Explain your reasoning by indicating whether the response is faithful to the information in the context, in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context.)

Judgment: (Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Response is largely unfaithful to the context, with unsupported information.
- 2: Insufficient – Some elements relate to the context, but there is unsupported information.
- 3: Passable – Relevant information, but with some inaccuracies.
- 4: Satisfactory – Mostly faithful, with a few missing details.
- 5: Very satisfactory – Fully faithful and complete according to the context.

Evaluation Guidelines:
- Verify if the response relies exclusively on the provided context without introducing external information.
- Ensure that the response faithfully reflects the main points of the context.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Here is the response to evaluate along with the provided context.

Response: {answer}
Context: {retrieved_contexts}
Response:::
"""

RETRIEVABILITY_EVAL_PROMPT = """
Evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Response:::
Evaluation: (Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.)

Judgment: (Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Context is mostly off-topic and lacks useful information.
- 2: Insufficient – Context is partially relevant, missing key information, with many irrelevant excerpts.
- 3: Acceptable – Context is generally relevant but diluted by several irrelevant excerpts.
- 4: Satisfactory – Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.
- 5: Very satisfactory – Context is entirely relevant and comprehensive, containing all necessary information.

Evaluation Guidelines:
- Check whether the context directly answers the question and if the excerpts are relevant to the response.
- Assess whether the presence of irrelevant excerpts affects clarity and comprehension.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Here is the question and the retrieved context for evaluation.

Question: {question}
Context: {retrieved_contexts}
Response:::
"""

CORRECTNESS_EVAL_PROMPT = """
Evaluate whether the response is correct, meaning it answers the question posed by providing essential information without significant factual errors.

Response:::
Evaluation: (Explain your reasoning for your judgment by indicating whether the response is correct, based on the question posed. Explicitly identify points of alignment or divergence with the question to support your judgment.)

Judgment: (Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Largely incorrect, with major errors.
- 2: Insufficient – Partially correct, with significant errors or inaccuracies.
- 3: Acceptable – Generally answers the question but contains several inaccuracies.
- 4: Satisfactory – Answers the question well, with only a few minor inaccuracies.
- 5: Very satisfactory – Completely correct, precise, and perfectly aligned with the question.

Evaluation Guidelines:
- Verify whether the response addresses all key aspects of the question without omissions.
- Ensure there are no misinterpretations or irrelevant information.
- Avoid penalizing the response for additional information that, while unnecessary, does not introduce errors or confusion.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Here is the question and the response for evaluation.

Question: {question}  
Response: {answer}  
Response:::
"""
                                                                ### Prometheus2 Eval Prompts ###


ABSOLUTE_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Score Rubrics:
{score_rubric}

###Feedback: """

CORRECTNESS_EVAL_PROMPT_PROMETHEUS = """
### Task Description:
A question, a response to evaluate, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback assessing the correctness of the response strictly based on the given score rubric, not in general.
2. After writing the feedback, assign a score to the response from 1 to 5 based on the rubric.
3. The output format should look as follows: "Feedback: (write feedback for correctness) [SCORE] (1-5)".
4. Please do not generate any other opening, closing, or explanations.

### Question:
{question}

### Response:
{answer}

### Score Rubric:
1: Largely incorrect, major errors.
2: Partially correct, significant inaccuracies.
3: Generally correct, several inaccuracies.
4: Correct, minor inaccuracies.
5: Fully correct and aligned with the question.

### Feedback:
"""

FAITHFULNESS_EVAL_PROMPT_PROMETHEUS = """
### Task Description:
A context, a question, a response to evaluate, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback assessing how faithfully the response aligns with the context and the given question, strictly based on the score rubric.
2. After writing the feedback, assign a score to the response from 1 to 5 based on the rubric.
3. The output format should look as follows: "Feedback: (write feedback for faithfulness) [SCORE] (1-5)".
4. Please do not generate any other opening, closing, or explanations.

### Context:
{retrieved_contexts}

### Question:
{question}

### Response:
{answer}

### Score Rubric:
1: Largely unfaithful, unsupported information.
2: Partially faithful, with inaccuracies.
3: Relevant, but with minor inaccuracies or extraneous details.
4: Mostly faithful, a few minor issues.
5: Fully faithful, no inaccuracies or unsupported details.

### Feedback:
"""

RETRIEVABILITY_EVAL_PROMPT_PROMETHEUS = """
### Task Description:
A question, a retrieved context, and a score rubric representing evaluation criteria are given.
1. Write detailed feedback assessing whether the context is relevant and sufficient to answer the question, strictly based on the given score rubric.
2. After writing the feedback, assign a score to the retrieved context from 1 to 5 based on the rubric.
3. The output format should look as follows: "Feedback: (write feedback for retrievability) [SCORE] (1-5)".
4. Please do not generate any other opening, closing, or explanations.

### Question:
{question}

### Context:
{retrieved_contexts}

### Score Rubric:
1: Mostly off-topic, lacks useful info.
2: Partially relevant, many irrelevant or missing key details.
3: Mostly relevant, some irrelevant parts.
4: Relevant, minor irrelevant parts.
5: Fully relevant and comprehensive.

### Feedback:
"""
