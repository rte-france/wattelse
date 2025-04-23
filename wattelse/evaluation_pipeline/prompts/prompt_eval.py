#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Dictionary-Based Prompts (Where you define your prompt for evaluation)

#################### Correctness evaluation prompts ####################
CORRECTNESS_EVAL_PROMPT = {
    "vanilla": """
You are a helpful assistant, please evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Evaluation: Explain your reasoning for your judgment by indicating whether the response is correct, based on the question asked. Explicitly identify points of alignment or divergence with the question to support your judgment. Do not penalize the response if it states that the documents do not provide specific information on this topic.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}  
Response: {answer}  
""",
    "vanilla-v2": """
You are a helpful assistant, please evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Evaluation: Explain your reasoning for your judgment by indicating whether the response is correct, based on the question asked. Explicitly identify points of alignment or divergence with the question to support your judgment. Do not penalize the response if it states that the documents do not provide specific information on this topic.
Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Largely incorrect, with major errors.
- 2: Insufficient – Partially correct, with significant errors or inaccuracies.
- 3: Acceptable – Generally answers the question but contains several inaccuracies.
- 4: Satisfactory – Answers the question well, with only a few minor inaccuracies.
- 5: Very satisfactory – Completely correct, precise, and perfectly aligned with the question.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}  
Response: {answer}  
""",
    "meta-llama-3-8b": """
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Response:::
Evaluation: (Explain your reasoning for your judgment by indicating whether the response is correct, based on the question asked. Explicitly identify points of alignment or divergence with the question to support your judgment.)

Judgment: Assign a score from 1 to 5 based on the following criteria:
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

Question: {question}  
Response: {answer}  
Response:::
""",
    "selene-mini": """
You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serves as the evaluation standard. Provide comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.  

Here are some rules of the evaluation:  
(1) Prioritize evaluating whether the response satisfies the provided rubric. The score should be based strictly on the rubric criteria. The response does not need to explicitly address all rubric points, but it should be assessed according to the outlined criteria.  

Your reply should strictly follow this format:  
**Reasoning:** <Your feedback>  

**Result:** <an integer between 1 and 5>  

### Here is the data:  

**Question:**  
{question}  

**Response:**  
{answer}  

**Instruction:**  
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.  

**Evaluation:**  
Indicate whether the response correctly answers the question, addressing all key aspects without omissions. Explicitly identify points of alignment or divergence with the question to support your judgment. Mention any factual errors, misinterpretations, or missing details that impact correctness.  

**Evaluation Guidelines:**  
- Verify whether the response addresses all key aspects of the question without omissions.  
- Ensure there are no misinterpretations or irrelevant information.  
- Avoid penalizing the response for additional information that, while unnecessary, does not introduce errors or confusion.  

**Score Rubrics:**  
[Evaluation of response correctness]  
- **Score 1:** Very insufficient – Largely incorrect, with major errors.  
- **Score 2:** Insufficient – Partially correct, with significant errors or inaccuracies.  
- **Score 3:** Acceptable – Generally answers the question but contains several inaccuracies.  
- **Score 4:** Satisfactory – Answers the question well, with only a few minor inaccuracies.  
- **Score 5:** Very satisfactory – Completely correct, precise, and perfectly aligned with the question.  
""",
    "deepseek": """
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Response:::
Evaluation:

Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Largely incorrect, with major errors.
- 2: Insufficient – Partially correct, with significant errors or inaccuracies.
- 3: Acceptable – Generally answers the question but contains several inaccuracies.
- 4: Satisfactory – Answers the question well, with only a few minor inaccuracies.
- 5: Very satisfactory – Completely correct, precise, and perfectly aligned with the question.

Evaluation Guidelines:
- Verify whether the response addresses all key aspects of the question without omissions.
- Ensure there are no misinterpretations or irrelevant information.
- Avoid penalizing the response for additional information that, while unnecessary, does not introduce errors or confusion.

You MUST provide values for 'Judgment:' in your response.

Here is the question:
Question: {question} 

Here is the response:
Response: {answer}  
Response:::
""",
}

#################### Faithfulness evaluation prompts ####################
FAITHFULNESS_EVAL_PROMPT = {
    "vanilla": """
You are a helpful assistant, please evaluate whether the response is based on the provided context, without introducing unsupported information.

Evaluation: Explain your reasoning by indicating whether the response is faithful to the information in the context, in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Response: {answer}  
Context: {retrieved_contexts}
""",
    "vanilla-v2": """
You are a helpful assistant, please evaluate whether the response is based on the provided context, without introducing unsupported information.

Evaluation: Explain your reasoning by indicating whether the response is faithful to the information in the context, in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context.
Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Response is largely unfaithful to the context, with unsupported information.
- 2: Insufficient – Some elements relate to the context, but there is unsupported information.
- 3: Passable – Relevant information, but with some inaccuracies.
- 4: Satisfactory – Mostly faithful, with a few missing details.
- 5: Very satisfactory – Fully faithful and complete according to the context.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Response: {answer}  
Context: {retrieved_contexts}
""",
    "meta-llama-3-8b": """
Evaluate whether the response is based on the provided context, without introducing unsupported information.

Response:::
Evaluation: (Explain your reasoning by indicating whether the response is faithful to the information in the context, in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context.)

Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Response is largely unfaithful to the context, with unsupported information.
- 2: Insufficient – Some elements relate to the context, but there is unsupported information.
- 3: Passable – Relevant information, but with some inaccuracies.
- 4: Satisfactory – Mostly faithful, with a few missing details.
- 5: Very satisfactory – Fully faithful and complete according to the context.

Evaluation Guidelines:
- Verify if the response relies exclusively on the provided context without introducing external information.
- Ensure that the response faithfully reflects the main points of the context.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Response: {answer}
Context: {retrieved_contexts}
Response:::
""",
    "selene-mini": """
You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serves as the evaluation standard. Provide comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.  

Here are some rules of the evaluation:  
(1) Prioritize evaluating whether the response satisfies the provided rubric. The score should be based strictly on the rubric criteria. The response does not need to explicitly address all rubric points, but it should be assessed according to the outlined criteria.  

Your reply should strictly follow this format:  
**Reasoning:** <Your feedback>  

**Result:** <an integer between 1 and 5>  

### Here is the data:  

**Response:**  
{answer}  

**Context:**  
{retrieved_contexts}  

**Instruction:**
Evaluate whether the response is factually accurate according to the provided context. The evaluation should focus on truth value rather than completeness.

**Evaluation:**  
Indicate whether the response is faithful to the provided context in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context. Specify whether missing details impact the completeness of the response.  

**Evaluation Guidelines:**
- Verify if each statement in the response is explicitly supported by the context
- Check if the response contains any claims that contradict the context
- Distinguish between missing information (which doesn't affect accuracy) and incorrect information

**Score Rubrics:**  
[Evaluation of response faithfulness]  
- **Score 1:** Incorrect – Response contains statements that directly contradict the context
- **Score 2:** Mostly incorrect – Response contains multiple statements not supported by the context
- **Score 3:** Mixed – Response contains both supported and unsupported statements
- **Score 4:** Mostly accurate – Response statements are supported by context with minor imprecisions
- **Score 5:** Completely accurate – All statements are directly supported by the context
""",
    "deepseek": """
Evaluate whether the response is based on the provided context, without introducing unsupported information.

Response:::
Evaluation:

Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Response is largely unfaithful to the context, with unsupported information.
- 2: Insufficient – Some elements relate to the context, but there is unsupported information.
- 3: Passable – Relevant information, but with some inaccuracies.
- 4: Satisfactory – Mostly faithful, with a few missing details.
- 5: Very satisfactory – Fully faithful and complete according to the context.

Evaluation Guidelines:
- Verify if the response relies exclusively on the provided context without introducing external information.
- Ensure that the response faithfully reflects the main points of the context.

You MUST provide values for 'Judgment:' in your response.

Here is the asnwer :
Response: {answer}

Here is the context :
Context: {retrieved_contexts}
Response:::
""",
}

#################### Retrievability evaluation prompts ####################

RETRIEVABILITY_EVAL_PROMPT = {
    "vanilla": """
You are a helpful assistant, please evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Evaluation: Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}
Context: {retrieved_contexts}
""",
    "vanilla-v2": """
You are a helpful assistant, please evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Evaluation: Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.
Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Context is mostly off-topic and lacks useful information.
- 2: Insufficient – Context is partially relevant, missing key information, with many irrelevant excerpts.
- 3: Acceptable – Context is generally relevant but diluted by several irrelevant excerpts.
- 4: Satisfactory – Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.
- 5: Very satisfactory – Context is entirely relevant and comprehensive, containing all necessary information.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}
Context: {retrieved_contexts}
""",
    "meta-llama-3-8b": """
Evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Response:::
Evaluation: (Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.)

Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Context is mostly off-topic and lacks useful information.
- 2: Insufficient – Context is partially relevant, missing key information, with many irrelevant excerpts.
- 3: Acceptable – Context is generally relevant but diluted by several irrelevant excerpts.
- 4: Satisfactory – Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.
- 5: Very satisfactory – Context is entirely relevant and comprehensive, containing all necessary information.

Evaluation Guidelines:
- Check whether the context directly answers the question and if the excerpts are relevant to the response.
- Assess whether the presence of irrelevant excerpts affects clarity and comprehension.

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}
Context: {retrieved_contexts}
Response:::
""",
    "deepseek": """
Evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Evaluation:

Judgment: Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Context is mostly off-topic and lacks useful information.
- 2: Insufficient – Context is partially relevant, missing key information, with many irrelevant excerpts.
- 3: Acceptable – Context is generally relevant but diluted by several irrelevant excerpts.
- 4: Satisfactory – Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.
- 5: Very satisfactory – Context is entirely relevant and comprehensive, containing all necessary information.

Evaluation Guidelines:
- Check whether the context directly answers the question and if the excerpts are relevant to the response.
- Assess whether the presence of irrelevant excerpts affects clarity and comprehension.

You MUST provide a score for 'Judgment:' in your response.

Here is the question :
Question: {question}

Here is the context :
Context: {retrieved_contexts}
""",
    "selene-mini": """
You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric that serves as the evaluation standard. Provide comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.  

Here are some rules of the evaluation:  
(1) Prioritize evaluating whether the response satisfies the provided rubric. The score should be based strictly on the rubric criteria. The response does not need to explicitly address all rubric points, but it should be assessed according to the outlined criteria.  

Your reply should strictly follow this format:  
**Reasoning:** <Your feedback>  

**Result:** <an integer between 1 and 5>  

### Here is the data:  

**Question:**  
{question}  

**Context:**  
{retrieved_contexts}  

**Instruction:**  
Evaluate whether the retrieved context is relevant and sufficient to answer the given question.  

**Evaluation:**  
Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.  

**Evaluation Guidelines:**  
- Check whether the context directly answers the question and if the excerpts are relevant.  
- Assess whether the presence of irrelevant excerpts affects clarity and comprehension.  

**Score Rubrics:**  
[Evaluation of context relevance and sufficiency]  
- **Score 1:** Context is mostly off-topic and lacks useful information.  
- **Score 2:** Context is partially relevant, missing key information, with many irrelevant excerpts.  
- **Score 3:** Context is generally relevant but diluted by several irrelevant excerpts.  
- **Score 4:** Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.  
- **Score 5:** Context is entirely relevant and comprehensive, containing all necessary information.  
""",
}

####################                                        ####################
#################### Pairwise Comparison Evaluation prompts ####################
####################                                        ####################

CORRECTNESS_PAIRWISE_PROMPT = {
    "default": """
You are a helpful assistant tasked with comparing two responses generated by different language models using a Retrieval-Augmented Generation (RAG) system.
Your goal is to determine which response better answers the given question.

Your evaluation should focus primarily on **correctness**, but may also take into account **clarity, tone, and helpfulness**, especially when these elements affect the user's ability to understand or apply the information.

### Evaluation Guidelines:
- Do **not** let the order of responses (Model 1 vs. Model 2) bias your decision. Either one could be better.
- Do **not** favor responses solely based on length. Evaluate whether the length is appropriate for answering the question effectively.
- Prioritize factual **correctness**, relevance, and alignment with the question and any implied context.
- Consider **stylistic elements** such as clarity, conciseness, coherence, and tone — but only in service of a better, more usable response.

---

**Question:**  
{question}

**Model 1 ({model1_name}) Answer:**  
{answer1}

**Model 2 ({model2_name}) Answer:**  
{answer2}

---

Provide a thoughtful comparison of both responses, highlighting factual correctness and any differences in clarity or presentation that influence the quality of the response. Your answer must follow this structure:

ANALYSIS:  
[Detailed comparison of both responses, focusing on correctness, relevance, clarity, and usability.]

WINNER: [Either "{model1_name}", "{model2_name}", or "Tie"]

REASON:  
[A concise summary justifying your choice, grounded in correctness but optionally noting stylistic advantages.]
"""
}


RETRIEVABILITY_PAIRWISE_PROMPT = {
    "default": """
You are an expert evaluator for retrieval systems. You will be given a question and two different sets of retrieved contexts.

Question: {question}

Model 1 ({model1_name}) Retrieved Context:
{context1}

Model 2 ({model2_name}) Retrieved Context:
{context2}

Evaluate both retrieved contexts ONLY on their relevance and completeness for answering the question.

Your response must follow this exact format:
ANALYSIS:
[Your detailed analysis of both contexts' relevance]

WINNER: [Either "{model1_name}", "{model2_name}", or "Tie"]

REASON:
[Brief explanation for your choice based on context quality]
""",
}

# Combine prompts in a nested dictionary
PROMPTS = {
    "correctness": CORRECTNESS_EVAL_PROMPT,
    "faithfulness": FAITHFULNESS_EVAL_PROMPT,
    "retrievability": RETRIEVABILITY_EVAL_PROMPT,
    "correctness_pairwise": CORRECTNESS_PAIRWISE_PROMPT,
    "retrievability_pairwise": RETRIEVABILITY_PAIRWISE_PROMPT,
}
