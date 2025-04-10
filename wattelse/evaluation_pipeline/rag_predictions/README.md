# RAG Predictions Engine

This tool generates predictions from a RAG pipeline using a dataset of questions and a document corpus.

## Files

* `rag_predictions.py` - Main script for generating RAG predictions

## Features

* Runs queries through a RAG pipeline
* Measures performance metrics (query time, retriever time)
* Records relevant extracts used by the RAG system
* Saves configurations with results

## Usage

```bash
python rag_predictions.py query_dataset.xlsx corpus_reference_dir --predictions-output-path output.xlsx
```

## Parameters

* `query_dataset.xlsx`: Dataset with questions and corpus reference (Excel format)
* `corpus_reference_dir`: Directory containing the corpus reference
* `--predictions-output-path`, `-o`: Output file path (optional)
* `--overwrite`: Overwrite existing output file if it exists (default: False)
  > ⚠️ **Warning**: Using this flag will delete any existing file at the output path without confirmation. Use with caution to avoid data loss.

## Output Format

The tool generates an Excel file with multiple sheets:
* `predictions`: Contains original questions with RAG answers and metrics
* `prompts_config`: System and user prompts configuration
* `retriever_config`: Embedding model and retrieval settings
* `generator_config`: LLM and generation parameters
* `collection_config`: Document collection information

## Configuration

Uses the predefined RAG configuration (`azure_20241216`):
* Document retrieval settings
* LLM generation parameters
* Custom prompt templates
* Performance tracking metrics