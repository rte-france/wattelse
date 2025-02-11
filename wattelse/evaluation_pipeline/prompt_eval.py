# Dictionary-Based Prompts (Where you define your prompt for evaluation)

                                                #################### Correctness evaluation prompts ####################
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


 "deepseek": """
Evaluate whether the response is correct, meaning it answers the question asked by providing essential information without significant factual errors.

Response:::
Evaluation:

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

You MUST provide values for 'Judgment:' in your response.

Here is the question:
Question: {question} 

Here is the response:
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
}

                                                #################### Faithfulness evaluation prompts ####################
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



    "deepseek": """
Evaluate whether the response is based on the provided context, without introducing unsupported information.

Response:::
Evaluation:

Judgment: (Assign a score from 1 to 5 based on the following criteria:
- 1: Very insufficient – Response is largely unfaithful to the context, with unsupported information.
- 2: Insufficient – Some elements relate to the context, but there is unsupported information.
- 3: Passable – Relevant information, but with some inaccuracies.
- 4: Satisfactory – Mostly faithful, with a few missing details.
- 5: Very satisfactory – Fully faithful and complete according to the context.

Evaluation Guidelines:
- Verify if the response relies exclusively on the provided context without introducing external information.
- Ensure that the response faithfully reflects the main points of the context.
- If the answer explicitly states that it couldn't find the information, it is not unfaithful.

You MUST provide values for 'Judgment:' in your response.

Here is the asnwer :
Response: {answer}

Here is the context :
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

                                                #################### Retrievability evaluation prompts ####################

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


    "deepseek": """
Evaluate whether the retrieved context is relevant and sufficient to answer the given question.

Evaluation:

Judgment: (Assign a score from 1 to 5 based on the following criteria:
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




    "selene-mini":"""
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

"""
}

# Combine prompts in a nested dictionary
PROMPTS = {
    "correctness": CORRECTNESS_EVAL_PROMPT,
    "faithfulness": FAITHFULNESS_EVAL_PROMPT,
    "retrievability": RETRIEVABILITY_EVAL_PROMPT,
}
