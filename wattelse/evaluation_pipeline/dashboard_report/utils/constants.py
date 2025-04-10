"""Constants used throughout the application."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

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
    "faithfulness": "Examines whether the response is based solely on the provided context without introducing unsupported information.",
    "correctness": "Assesses whether the response correctly answers the question by providing essential information without significant factual errors.",
    "retrievability": "Determines whether the retrieved context is relevant and sufficient to answer the given question.",
}
