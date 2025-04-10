# Synthetic QA Generation

This tool generates synthetic question-answer triplets from documents for RAG evaluation.

## Files

* `generate_qa.py` - Main script for generating synthetic QA triplets
* `prompt_QA_gen.py` - Contains the prompt template for QA generation

## Features

* Generates three question types:
  * Simple factual questions
  * Reasoning questions
  * Multi-context questions
* Processes document chunks efficiently
* Outputs results to Excel format

## Usage

```bash
python generate_qa.py --eval-corpus-path data/eval_corpus --n-generations 100 --output-path qa_output.xlsx
```

## Parameters

* `--eval-corpus-path`: Input document directory
* `--n-generations`: Number of QA triplets to generate (default: 100)
* `--output-path`: Output file path (default: qa_output.xlsx)

## Output Format

The tool generates an Excel file with:
* Document context
* Generated questions and answers
* Question complexity type
* Source document information

## Configuration

Default settings:
* 1 document per QA generation
* 10 chunks per document 
* Maximum tokens: 60000