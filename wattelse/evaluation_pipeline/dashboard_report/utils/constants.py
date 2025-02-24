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

# Metric descriptions for UI
METRIC_DESCRIPTIONS = {
    'faithfulness': 'Assesses how well the response relies on the provided context, ensuring that every claim is directly supported by that context without introducing any external or unsupported details.',
    'correctness': 'Evaluates whether the response factually and accurately addresses the question by providing all essential information without significant errors or misinterpretations.',
    'retrievability': 'Evaluates the relevance and sufficiency of the retrieved context in answering the question, considering both the presence of key information and the impact of any irrelevant excerpts on overall clarity.'
}