"""Constants used throughout the application."""

# Column definitions
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "source_doc"
CONTEXT_COLUMN = "context"
COMPLEXITY_COLUMN = "complexity"
RAG_RELEVANT_EXTRACTS_COLUMN = "relevant_extracts"
RAG_QUERY_TIME_COLUMN = "rag_query_time_seconds"
RAG_RETRIEVER_TIME_COLUMN = "rag_retriever_time_seconds"

# Define a consistent color scheme for judges
JUDGE_COLORS = {
    'Meta-Llama-3-8B-Instruct': '#1f77b4',  # Blue
    'prometheus-7b-v2.0': '#2ca02c',        # Green
    'Selene-1-Mini-Llama-3.1-8B': '#ff7f0e', # Orange
    'DeepSeek-R1-Distill-Llama-8B': '#d62728' # Red
}

# Metric descriptions for UI
METRIC_DESCRIPTIONS = {
    'faithfulness': 'Measures how well the answer aligns with the provided context',
    'correctness': 'Evaluates the factual accuracy of the response',
    'retrievability': 'Assesses the relevance of retrieved documents'
}