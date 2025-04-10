# Prompts

This directory contains prompt templates and response parsing logic for the RAG evaluation pipeline.

## Files

- `prompt_eval.py` - Defines evaluation prompts for different metrics and models
- `regex_patterns.py` - Contains regex patterns for parsing responses from different models

## Prompt Templates

The system uses different prompt templates for:

1. **Correctness Evaluation** - Assessing if responses correctly answer questions
2. **Faithfulness Evaluation** - Measuring if responses stick to provided context
3. **Retrievability Evaluation** - Evaluating the quality of retrieved context

Each metric has model-specific prompt variations to account for different LLM capabilities and formats.

## Response Parsing

The `RegexPatterns` class provides model-specific regex patterns for extracting:
- Evaluation text (reasoning and explanations)
- Judgment scores (numerical ratings)

This allows standardized parsing of responses from different LLM architectures which may format their outputs differently.

## Example Prompt Structure

```python
CORRECTNESS_EVAL_PROMPT = {
    "meta-llama-3-8b": """
    Evaluate whether the response is correct, meaning it answers the question asked...
    
    Response:::
    Evaluation: (Explain your reasoning...)
    
    Judgment: Assign a score from 1 to 5 based on the following criteria:
    - 1: Very insufficient – Largely incorrect, with major errors.
    ...
    
    Question: {question}  
    Response: {answer}  
    Response:::
    """,
    
    # Other model-specific prompts...
}
```

## Usage

```python
from wattelse.evaluation_pipeline.prompts.prompt_eval import PROMPTS
from wattelse.evaluation_pipeline.prompts.regex_patterns import RegexPatterns

# Get prompt templates
correctness_prompts = PROMPTS["correctness"]

# Get model-specific prompt
llama_prompt = correctness_prompts["meta-llama-3-8b"]

# Get regex patterns for parsing
patterns = RegexPatterns().get_patterns("re_llama3")
```