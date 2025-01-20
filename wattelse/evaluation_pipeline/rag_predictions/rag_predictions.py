#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import re
import time
import typer
import pandas as pd
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from wattelse.chatbot.backend.rag_backend import RAGBackEnd

#TODO Parellelize (optional)
# from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib

QUERY_COLUMN = "question"
DOC_LIST_COLUMN = "source_doc"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"
RAG_QUERY_TIME_COLUMN = "rag_query_time_seconds"
SPECIAL_CHARACTER_FILTER = (
    r"[\t\n\r\x07\x08\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009]"
)

app = typer.Typer()

@app.command()
def main(
    qr_df_path: Path,
    eval_corpus_path: Path,
    predictions_output_path: Path = Path(__file__).parent / "predictions_output.xlsx"  # Default RAG output path
):
    """
    python rag_predictions.py <qr_df_path> <eval_corpus_path> --report-output-path <output_path>

    Main function to generate predictions from the RAG pipeline.
    This function retrieves answers using RAG without performing any evaluation.

    Args:
        qr_df_path (Path): Path to the query and answer dataset (in Excel format).
        eval_corpus_path (Path): Path to the evaluation corpus folder.
        predictions_output_path (Path, optional): Path to save the predictions (Excel file). Default is 'predictions_output.xlsx'.
    """

    # Initialize RAG backend
    eval_group_id = "rag_eval"
    RAGBackEnd(eval_group_id).clear_collection()  # Ensure RAG backend is empty
    RAG_EVAL_BACKEND = RAGBackEnd(eval_group_id)
    logger.info(f"RAG Backend initialized with LLM: {RAG_EVAL_BACKEND.llm.model_name}")

    # Load data
    eval_df = pd.read_excel(qr_df_path)

    # Transform source_doc to a list by splitting on commas and trimming whitespace for each document name
    eval_df[DOC_LIST_COLUMN] = eval_df[DOC_LIST_COLUMN].apply(
        lambda x: [doc.strip() for doc in x.split(",")] if pd.notnull(x) else []
    )

    # Flatten and deduplicate all document names
    all_doc_list = set(
        [item for sublist in eval_df[DOC_LIST_COLUMN].to_list() for item in sublist]
    )

    all_eval_corpus_files = set([doc.name for doc in eval_corpus_path.iterdir()])
    for doc in all_doc_list:
        if doc not in all_eval_corpus_files:
            raise ValueError(f"Document {doc} not found in eval corpus folder")

    # Load eval docs in RAG backend
    for doc in tqdm(eval_corpus_path.iterdir(), desc="Loading documents into RAG Backend"):
        if doc.name in all_doc_list:
            with open(doc, "rb") as f:
                RAG_EVAL_BACKEND.add_file_to_collection(doc.name, f)
                logger.info(f"Added {doc.name} to collection")

    # Get RAG predictions
    rag_answers = []
    rag_relevant_extracts = []
    rag_query_times = []
    
    for _, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Getting RAG Predictions"):
        logger.info(f"Processing row: question={row[QUERY_COLUMN]}")

        # Measure query time
        start_time = time.time()
        response = RAG_EVAL_BACKEND.query_rag(
            row[QUERY_COLUMN], selected_files=row[DOC_LIST_COLUMN]
        )
        query_time = time.time() - start_time
        
        answer = response.get("answer", "")
        relevant_extracts = [
            re.sub(SPECIAL_CHARACTER_FILTER, " ", extract["content"])
            for extract in response.get("relevant_extracts", [])
        ]

        logger.info(f"RAG Answer for question '{row[QUERY_COLUMN]}': {answer}")
        logger.info(f"RAG Relevant Extracts: {relevant_extracts}")
        logger.info(f"Query time: {query_time:.2f} seconds")

        rag_answers.append(answer)
        rag_relevant_extracts.append(relevant_extracts)
        rag_query_times.append(query_time)

    # Save predictions to a DataFrame
    eval_df["rag_answer"] = rag_answers
    eval_df[RAG_RELEVANT_EXTRACTS_COLUMN] = [
        "\n\n".join([f"Extrait {i + 1}: {extract}" for i, extract in enumerate(sublist)])
        for sublist in rag_relevant_extracts
    ]
    eval_df[RAG_QUERY_TIME_COLUMN] = rag_query_times

    # Add summary statistics for query times
    logger.info(f"Average query time: {pd.Series(rag_query_times).mean():.2f} seconds")
    logger.info(f"Maximum query time: {pd.Series(rag_query_times).max():.2f} seconds")
    logger.info(f"Minimum query time: {pd.Series(rag_query_times).min():.2f} seconds")

    # Save the predictions to an Excel file
    eval_df.to_excel(predictions_output_path, index=False)
    logger.info(f"Predictions saved to {predictions_output_path}")

    # Clear RAG backend
    RAG_EVAL_BACKEND.clear_collection()

if __name__ == "__main__":
    typer.run(main)