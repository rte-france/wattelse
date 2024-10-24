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
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_GROUNDEDNESS_PROMPT,
)

# Example :
# python main.py QA_GENE.xlsx data/test_gen --report-output-path data/eval_output_update-Precision20.xlsx

# Updated column names to match new format
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "source_doc"
CONTEXT_COLUMN = "context"
COMPLEXITY_COLUMN = "complexity"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"

# Characters to be removed from relevant extracts for better readability
SPECIAL_CHARACTER_FILTER = (
    r"[\t\n\r\x07\x08\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009]"
)

# Define the Typer app
app = typer.Typer()

def call_llm(llm_client, prompt: str):
    """Function to call the LLM with the given prompt."""
    response = llm_client.generate(prompt, temperature=0)
    return response

def evaluate_metrics(llm_client, question, context_extracted):
    """Function to evaluate multiple metrics using the LLM."""
    total_chunks = len(context_extracted.split("\n\n"))

    # Evaluate precision
    precision_eval = call_llm(llm_client, CONTEXT_PRECISION_PROMPT.format(
        question=question,
        retrieved_contexts=context_extracted,
        total_chunks=total_chunks
    ))

    # Evaluate groundedness
    groundedness_eval = call_llm(llm_client, CONTEXT_GROUNDEDNESS_PROMPT.format(
        retrieved_contexts=context_extracted,
        question=question,
    ))

    evaluations = {}

    # Process precision evaluation
    logger.debug(f"Precision LLM response: {precision_eval}")
    # Updated regex to capture both integer and decimal numbers
    precision_score_line = precision_eval.split("Précision : ")[-1].split("\n")[0].strip()
    precision_score_match = re.search(r"(\d+([,.]\d+)?)", precision_score_line)  # Updated regex to capture integers like "0" and decimals

    evaluations["precision_evaluation"] = precision_eval
    # Extract precision score, handling both integers and decimals
    evaluations["precision_score"] = float(precision_score_match.group(1).replace(',', '.')) if precision_score_match else np.nan

    logger.debug(f"Groundedness LLM response: {groundedness_eval}")
    # Extract evaluations and scores for groundedness
    evaluations["groundedness"] = groundedness_eval.split("Évaluation : ")[-1].split("Note totale :")[0].strip()
    evaluations["groundedness_score"] = groundedness_eval.split("Note totale :")[-1].strip()

    return evaluations


def evaluate_rag_metrics(eval_df):
    llm_client = OpenAI_Client()   # Initialize the LLM client for critique
    logger.info(f"LLM Evaluation model: {llm_client.model_name}")

    evaluations = []

    # Use tqdm for progress tracking
    for idx, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Evaluating Multiple Metrics"):
        try:
            question = row[QUERY_COLUMN]
            context_extracted = row[RAG_RELEVANT_EXTRACTS_COLUMN]

            # Ensure context is not empty
            if not context_extracted.strip():
                logger.warning(f"Empty context for question {idx}: {question}")
                evaluations.append({
                    "question": question,
                    "evaluation": "No context provided",
                    "precision_score": "No context provided",
                    "groundedness": "No context provided",
                    "groundedness_score": "No context provided"
                })
                continue

            logger.debug(f"Evaluating metrics for question {idx}: {question}")

            # Evaluate all metrics
            eval_results = evaluate_metrics(llm_client, question, context_extracted)

            # Add question and results to evaluation list
            eval_entry = {
                "question": question
            }
            eval_entry.update(eval_results)
            evaluations.append(eval_entry)

            logger.info(f"Evaluations for question {idx}: {eval_results}")

        except Exception as e:
            logger.error(f"Error evaluating metrics for row {idx}: {e}")
            evaluations.append({
                "question": row[QUERY_COLUMN],
                "evaluation": "Error during evaluation",
                "precision_score": "Error during evaluation",
                "groundedness": "Error during evaluation",
                "groundedness_score": "Error during evaluation",
            })

    # Convert evaluations to DataFrame
    evaluation_df = pd.DataFrame(evaluations)
    eval_df = eval_df.join(evaluation_df.set_index("question"), on=QUERY_COLUMN, rsuffix="_eval")
    return eval_df

def extract_numeric(score):
    """Function to extract numeric values from score strings, handling 0 correctly."""
    try:
        # Extract the numeric part of the score
        numeric_score = re.search(r"(\d+(\.\d+)?)", str(score))
        if numeric_score:
            return float(numeric_score.group(1))
        else:
            return np.nan
    except Exception:
        return np.nan


@app.command()
def main(
    qr_df_path: Path,
    eval_corpus_path: Path,
    report_output_path: Path = Path(__file__).parent / "report_output.xlsx"  # Default RAG output path
):
    """
    Function to evaluate the generation part of the RAG pipeline.
    Currently supports multiple metrics (Precision, Groundedness).
    """
    # Initialize RAG backend and LLM client
    eval_group_id = "rag_eval"
    RAGBackEnd(eval_group_id).clear_collection()  # Ensure RAG eval backend is empty
    RAG_EVAL_BACKEND = RAGBackEnd(eval_group_id)
    logger.info(f"RAG Backend LLM: {RAG_EVAL_BACKEND.llm.model_name}")

    # Load data
    eval_df = pd.read_excel(qr_df_path)

    # Transform source_doc to list for processing
    eval_df[DOC_LIST_COLUMN] = eval_df[DOC_LIST_COLUMN].apply(lambda x: [x] if pd.notnull(x) else [])

    # Check all documents listed in `doc_list` are present in `eval_corpus_path` folder
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
    for _, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Getting RAG Predictions"):
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
        "\n\n".join([f"Extrait {i + 1}: {extract}" for i, extract in enumerate(sublist)])
        for sublist in rag_relevant_extracts
    ]

    # Evaluate multiple metrics using LLM and save results
    evaluated_df = evaluate_rag_metrics(eval_df)

    output_data = []
    for idx, row in evaluated_df.iterrows():
        output_data.append({
            "context": row[CONTEXT_COLUMN],
            "question": row[QUERY_COLUMN],
            "answer": row["rag_answer"],
            "source_doc": row[DOC_LIST_COLUMN],
            "relevant_extracts": row[RAG_RELEVANT_EXTRACTS_COLUMN], 
            "groundedness_evaluation": row.get("groundedness"),
            "groundedness_score": row.get("groundedness_score"),
            "precision_evaluation": row.get("precision_evaluation"),
            "precision_score": row.get("precision_score"),
        })


    # Save the final evaluated QA pairs to an Excel file
    df_output = pd.DataFrame(output_data)

    # Apply the function to the score columns
    df_output['groundedness_score'] = df_output['groundedness_score'].apply(extract_numeric)
    # df_output['precision_score'] = df_output['precision_score'].apply(extract_numeric)

    df_output.to_excel(report_output_path, index=False)
    print(f"Final evaluated QA dataset saved to {report_output_path}")

    # Clear RAG backend
    RAG_EVAL_BACKEND.clear_collection()

if __name__ == "__main__":
    typer.run(main)
