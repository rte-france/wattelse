#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import re
import typer
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.chatbot.eval.prompt import (
    FAITHFULNESS_EVAL_PROMPT,
    CORRECTNESS_EVAL_PROMPT,
    RETRIEVABILITY_EVAL_PROMPT,
    # CONTEXT_NDCG_PROMPT  # Testing Phase
)
from joblib import Parallel, delayed  # Import Parallel and delayed for parallelization
from tqdm_joblib import tqdm_joblib  # Import the tqdm_joblib helper

# Example :
# python main.py data/QA_Corpus-A.xlsx data/eval_corpus_A --report-output-path data/eval_output_update_Corpus_A.xlsx

# Updated column names to match new format
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "source_doc"
CONTEXT_COLUMN = "context"
COMPLEXITY_COLUMN = "complexity"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"
SPECIAL_CHARACTER_FILTER = (
    r"[\t\n\r\x07\x08\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009]"
)

app = typer.Typer()


def call_llm(llm_client, prompt: str) -> str:
    """Function to call the LLM with the given prompt."""
    response = llm_client.generate(prompt, temperature=0)
    return response


def evaluate_metrics(llm_client, question, answer, context_extracted) -> dict:
    """
    Evaluates the answer based on multiple metrics (faithfulness, correctness, retrievability) using the LLM.

    Args:
        llm_client (OpenAI_Client): The client to interact with the OpenAI LLM.
        question (str): The question to evaluate.
        answer (str): The answer to evaluate.
        context_extracted (str): The context used to generate the answer.

    Returns:
        dict: A dictionary containing the evaluations and scores for faithfulness, correctness, and retrievability.
    """

    # Evaluate faithfulness
    faithfulness_eval = call_llm(
        llm_client,
        FAITHFULNESS_EVAL_PROMPT.format(
            retrieved_contexts=context_extracted, answer=answer
        ),
    )

    # Evaluate correctness
    correctness_eval = call_llm(
        llm_client, CORRECTNESS_EVAL_PROMPT.format(question=question, answer=answer)
    )

    # Evaluate retrievability
    retrievability_eval = call_llm(
        llm_client,
        RETRIEVABILITY_EVAL_PROMPT.format(
            question=question, retrieved_contexts=context_extracted
        ),
    )

    evaluations = {}

    # Parse evaluations
    logger.debug(f"faithfulness LLM response: {faithfulness_eval}")
    evaluation_match = re.search(r"Évaluation :\s*(.*)", faithfulness_eval, re.DOTALL)
    score_match = re.search(r"Jugement :\s*([1-5])", faithfulness_eval)
    evaluations["faithfulness"] = (
        evaluation_match.group(1).strip() if evaluation_match else "Not provided"
    )
    evaluations["faithfulness_score"] = (
        int(score_match.group(1)) if score_match else np.nan
    )

    logger.debug(f"correctness LLM response: {correctness_eval}")
    correctness_eval_match = re.search(
        r"Évaluation :\s*(.*)", correctness_eval, re.DOTALL
    )
    correctness_score_match = re.search(r"Jugement :\s*([1-5])", correctness_eval)
    evaluations["correctness"] = (
        correctness_eval_match.group(1).strip()
        if correctness_eval_match
        else "Not provided"
    )
    evaluations["correctness_score"] = (
        int(correctness_score_match.group(1)) if correctness_score_match else np.nan
    )

    logger.debug(f"retrievability LLM response: {retrievability_eval}")
    retrievability_eval_match = re.search(
        r"Évaluation :\s*(.*)", retrievability_eval, re.DOTALL
    )
    retrievability_score_match = re.search(r"Jugement :\s*([1-5])", retrievability_eval)
    evaluations["retrievability"] = (
        retrievability_eval_match.group(1).strip()
        if retrievability_eval_match
        else "Not provided"
    )
    evaluations["retrievability_score"] = (
        int(retrievability_score_match.group(1))
        if retrievability_score_match
        else np.nan
    )

    return evaluations


def evaluate_rag_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the generated answers from the RAG (retrieval-augmented generation) pipeline using multiple metrics.

    Args:
        eval_df (pandas.DataFrame): The DataFrame containing the evaluation corpus and answers to evaluate.

    Returns:
        pandas.DataFrame: The DataFrame with added evaluation columns (faithfulness, correctness, retrievability).
    """
    logger.info(f"LLM Evaluation model: {OpenAI_Client().model_name}")

    # Wrap the Parallel execution with tqdm_joblib to show progress
    with tqdm_joblib(desc="Evaluating Rows", total=eval_df.shape[0]) as progress_bar:
        evaluations = Parallel(n_jobs=-1)(
            delayed(evaluate_row)(row)
            for _, row in eval_df.iterrows()  # Unpack tuple here
        )

    # Combine evaluations into a DataFrame
    evaluation_df = pd.DataFrame(evaluations)

    # Join the evaluations to the original DataFrame
    eval_df = eval_df.join(
        evaluation_df.set_index("question"), on=QUERY_COLUMN, rsuffix="_eval"
    )

    return eval_df


def evaluate_row(row) -> dict:
    """Function to evaluate a single row of data (question, context, and answer)."""
    llm_client = OpenAI_Client()  # Initialize the LLM client inside the worker function
    try:
        question = row[QUERY_COLUMN]
        context_extracted = row[RAG_RELEVANT_EXTRACTS_COLUMN]
        answer = row["rag_answer"]

        if not context_extracted.strip():
            logger.warning(f"Empty context for question: {question}")
            return {
                "question": question,
                "evaluation": "No context provided",
                "faithfulness_score": "No context provided",
                "correctness_score": "No context provided",
                "retrievability_score": "No context provided",
            }

        eval_results = evaluate_metrics(llm_client, question, context_extracted, answer)

        eval_entry = {"question": question}
        eval_entry.update(eval_results)
        logger.info(f"Evaluations for question: {question}: {eval_results}")
        return eval_entry

    except Exception as e:
        logger.error(f"Error evaluating metrics: {e}")
        return {
            "question": row[QUERY_COLUMN],
            "evaluation": "Error during evaluation",
            "faithfulness_score": "Error during evaluation",
            "correctness_score": "Error during evaluation",
            "retrievability_score": "Error during evaluation",
        }


@app.command()
def main(
    qr_df_path: Path,
    eval_corpus_path: Path,
    report_output_path: Path = Path(__file__).parent
    / "report_output.xlsx",  # Default RAG output path
):
    """
    Main function to evaluate the generation part of the RAG pipeline.
    This function computes multiple metrics such as faithfulness, correctness, and retrievability for the RAG answers.

    Args:
        qr_df_path (Path): Path to the query and answer dataset (in Excel format).
        eval_corpus_path (Path): Path to the evaluation corpus folder.
        report_output_path (Path, optional): Path to save the evaluation report (Excel file). Default is 'report_output.xlsx'.

    """

    # Initialize RAG backend and LLM client
    eval_group_id = "rag_eval"
    RAGBackEnd(eval_group_id).clear_collection()  # Ensure RAG eval backend is empty
    RAG_EVAL_BACKEND = RAGBackEnd(eval_group_id)
    logger.info(f"RAG Backend LLM: {RAG_EVAL_BACKEND.llm.model_name}")

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
    for doc in tqdm(
        eval_corpus_path.iterdir(), desc="Loading documents into RAG Backend"
    ):
        if doc.name in all_doc_list:
            with open(doc, "rb") as f:
                RAG_EVAL_BACKEND.add_file_to_collection(doc.name, f)
                logger.info(f"Added {doc.name} to collection")

    # Get RAG predictions
    rag_answers = []
    rag_relevant_extracts = []
    for _, row in tqdm(
        eval_df.iterrows(), total=eval_df.shape[0], desc="Getting RAG Predictions"
    ):
        logger.info(f"Processing row: context={row[QUERY_COLUMN]}")

        response = RAG_EVAL_BACKEND.query_rag(
            row[QUERY_COLUMN], selected_files=row[DOC_LIST_COLUMN]
        )
        answer = response.get(ANSWER_COLUMN, "")
        relevant_extracts = [
            re.sub(SPECIAL_CHARACTER_FILTER, " ", extract["content"])
            for extract in response.get("relevant_extracts", [])
        ]

        logger.info(f"RAG Answer for question '{row[QUERY_COLUMN]}': {answer}")
        logger.info(f"RAG Relevant Extracts: {relevant_extracts}")

        rag_answers.append(answer)
        rag_relevant_extracts.append(relevant_extracts)

    eval_df["rag_answer"] = rag_answers
    eval_df[RAG_RELEVANT_EXTRACTS_COLUMN] = [
        "\n\n".join(
            [f"Extrait {i + 1}: {extract}" for i, extract in enumerate(sublist)]
        )
        for sublist in rag_relevant_extracts
    ]

    # Evaluate multiple metrics using LLM and save results
    evaluated_df = evaluate_rag_metrics(eval_df)

    output_data = []
    for _, row in evaluated_df.iterrows():
        output_data.append(
            {
                "context": row[CONTEXT_COLUMN],
                "question": row[QUERY_COLUMN],
                "answer": row["rag_answer"],
                "complexity": row[COMPLEXITY_COLUMN],
                "source_doc": row[DOC_LIST_COLUMN],
                "relevant_extracts": row[RAG_RELEVANT_EXTRACTS_COLUMN],
                "faithfulness_evaluation": row.get("faithfulness"),
                "faithfulness_score": row.get("faithfulness_score"),
                "correctness_evaluation": row.get("correctness"),
                "correctness_score": row.get("correctness_score"),
                "retrievability_evaluation": row.get("retrievability"),
                "retrievability_score": row.get("retrievability_score"),
            }
        )

    # Save the final evaluated QA pairs to an Excel file
    df_output = pd.DataFrame(output_data)

    # Convert lists in 'source_doc' to a string before dropping duplicates
    df_output["source_doc"] = df_output["source_doc"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else x
    )

    # Drop duplicates based on 'question' and 'source_doc'
    df_output = df_output.drop_duplicates(subset=["question", "source_doc"])

    # Calculate averages
    df_output["faithfulness_score"] = pd.to_numeric(
        df_output["faithfulness_score"], errors="coerce"
    )
    df_output["correctness_score"] = pd.to_numeric(
        df_output["correctness_score"], errors="coerce"
    )
    df_output["retrievability_score"] = pd.to_numeric(
        df_output["retrievability_score"], errors="coerce"
    )

    average_faithfulness = df_output["faithfulness_score"].mean()
    average_correctness = df_output["correctness_score"].mean()
    average_retrievability = df_output["retrievability_score"].mean()

    # logger.info averages
    logger.info(f"Average faithfulness: {average_faithfulness:.2f}")
    logger.info(f"Average correctness: {average_correctness:.2f}")
    logger.info(f"Average retrievability: {average_retrievability:.2f}")

    # Save the final evaluated QA pairs to an Excel file
    df_output.to_excel(report_output_path, index=False)
    logger.info(f"Final evaluated QA dataset saved to {report_output_path}")

    # Clear RAG backend
    RAG_EVAL_BACKEND.clear_collection()


if __name__ == "__main__":
    typer.run(main)
