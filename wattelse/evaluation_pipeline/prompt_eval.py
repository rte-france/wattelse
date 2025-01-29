# Dictionary-Based Prompts (Where you define your prompt for evaluation)

# Correctness evaluation prompts
CORRECTNESS_EVAL_PROMPT = {
    "default": """
You are a helpful assistant, please evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Evaluation: Explain your reasoning for your judgment by indicating whether the response is correct, based on the question asked. Explicitly identify points of alignment or divergence with the question to support your judgment.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}  
Response: {answer}  
""",
 "meta-llama-3-8b": """
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Response:::
Evaluation: (Explain your reasoning for your judgment by indicating whether the response is correct, based on the question asked. Explicitly identify points of alignment or divergence with the question to support your judgment.)

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

Question: {question}  
Response: {answer}  
Response:::
""",
    "prometheus": """
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
""",
    "skywork":"""
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.
Your evaluation should consider:

Alignment: Does the response address all aspects of the question directly?
Accuracy: Are there factual errors or inaccuracies?
Completeness: Does the response cover the essential details without omitting critical parts of the question?
Do not penalize for additional but accurate information provided it that does not introduce confusion. Use the following criteria to assign a Judgment score from 1 to 5:

1: Very insufficient – Largely incorrect, with major errors.
2: Insufficient – Partially correct, with significant errors or inaccuracies.
3: Acceptable – Generally answers the question but contains several inaccuracies.
4: Satisfactory – Answers the question well, with only a few minor inaccuracies.
5: Very satisfactory – Completely correct, precise, and perfectly aligned with the question.
You must provide:

Evaluation: Explain your reasoning based on correctness, citing specific alignment or errors.
Judgment: Assign a score (1–5).

Structure for evaluation:

Question: {question}
Response: {answer}

Response:::
Evaluation: (Explain reasoning here.)
Judgment: (Score from 1–5)
"""
}

# Faithfulness evaluation prompts
FAITHFULNESS_EVAL_PROMPT = {
    "default": """
You are a helpful assistant, please evaluate whether the response is based on the provided context, without introducing unsupported information.

Evaluation: Explain your reasoning by indicating whether the response is faithful to the information in the context, in terms of relevance and sufficiency. Explicitly identify points of alignment or divergence with the context.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Response: {answer}  
Context: {retrieved_contexts}
""",
    "meta-llama-3-8b": """
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

Response: {answer}
Context: {retrieved_contexts}
Response:::
""",
    "prometheus": """
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
}

# Retrievability evaluation prompts
RETRIEVABILITY_EVAL_PROMPT = {
    "default": """
You are a helpful assistant, please evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Evaluation: Indicate whether the context allows the question to be answered and contains the necessary information. Specify if the proportion of irrelevant excerpts compared to the total impacts the quality of the response, and mention any lack of completeness.
Judgment: (Assign a score from 1 to 5)

You MUST provide values for 'Evaluation:' and 'Judgment:' in your response.

Question: {question}
Context: {retrieved_contexts}
""",
    "meta-llama-3-8b": """
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

Question: {question}
Context: {retrieved_contexts}
Response:::
""",
    "prometheus": """
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
""",
    "skywork":"""
Evaluate whether the retrieved context is relevant and sufficient to answer the given question. Your evaluation should consider:

Relevance: Does the context align with the question and response?
Sufficiency: Does the context provide all necessary information to address the question completely?
Irrelevance: Consider the proportion of irrelevant excerpts in the context and their impact on clarity and comprehension.
Use the following criteria to assign a Judgment score from 1 to 5:

1: Very insufficient – Context is mostly off-topic and lacks useful information.
2: Insufficient – Context is partially relevant, missing key information, with many irrelevant excerpts.
3: Acceptable – Context is generally relevant but diluted by several irrelevant excerpts.
4: Satisfactory – Context is mostly relevant, with only a few irrelevant excerpts that do not strongly affect comprehension.
5: Very satisfactory – Context is entirely relevant and comprehensive, containing all necessary information.
You must provide:

Evaluation: Explain whether the context directly supports answering the question, citing any issues like irrelevance or insufficiency.
Judgment: Assign a score (1–5).
Structure for evaluation:

Question: {question}
Context: {retrieved_contexts}

Response:::
Evaluation: (Explain reasoning here.)
Judgment: (Score from 1–5
"""
}

# Combine prompts in a nested dictionary
PROMPTS = {
    "correctness": CORRECTNESS_EVAL_PROMPT,
    "faithfulness": FAITHFULNESS_EVAL_PROMPT,
    "retrievability": RETRIEVABILITY_EVAL_PROMPT,
}
