import re
import typer
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.chatbot.eval.prompt import (
    FAITHFULNESS_EVAL_PROMPT_PROMETHEUS,
    RETRIEVABILITY_EVAL_PROMPT_PROMETHEUS,
    CORRECTNESS_EVAL_PROMPT_PROMETHEUS,
)
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

# Example :
# python evaluations.py data/first_LLM_eval.xlsx --report-output-path data/second_LLM_eval.xlsx

# Column definitions
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "source_doc"
CONTEXT_COLUMN = "context"
COMPLEXITY_COLUMN = "complexity"
RAG_RELEVANT_EXTRACTS_COLUMN = "relevant_extracts"
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
    """
    # Evaluate faithfulness
    faithfulness_eval = call_llm(llm_client, FAITHFULNESS_EVAL_PROMPT_PROMETHEUS.format(
        retrieved_contexts=context_extracted,
        answer=answer,
        question=question
    ))
    
    # Evaluate correctness
    correctness_eval = call_llm(llm_client, CORRECTNESS_EVAL_PROMPT_PROMETHEUS.format(
        question=question,
        answer=answer
    ))
    
    # Evaluate retrievability
    retrievability_eval = call_llm(llm_client, RETRIEVABILITY_EVAL_PROMPT_PROMETHEUS.format(
        question=question,
        retrieved_contexts=context_extracted
    ))

    evaluations = {}

    # Parse evaluations
    logger.debug(f"faithfulness LLM response: {faithfulness_eval}")
    evaluation_match = re.search(r"(.*?)(?=\[SCORE\])", faithfulness_eval, re.DOTALL)
    score_match = re.search(r"\[SCORE\]\s*([1-5])", faithfulness_eval)
    evaluations["faithfulness"] = evaluation_match.group(1).strip() if evaluation_match else faithfulness_eval
    evaluations["faithfulness_score"] = int(score_match.group(1)) if score_match else np.nan

    logger.debug(f"correctness LLM response: {correctness_eval}")
    correctness_eval_match = re.search(r"(.*?)(?=\[SCORE\])", correctness_eval, re.DOTALL)
    correctness_score_match = re.search(r"\[SCORE\]\s*([1-5])", correctness_eval)
    evaluations["correctness"] = correctness_eval_match.group(1).strip() if correctness_eval_match else correctness_eval
    evaluations["correctness_score"] = int(correctness_score_match.group(1)) if correctness_score_match else np.nan

    logger.debug(f"retrievability LLM response: {retrievability_eval}")
    retrievability_eval_match = re.search(r"(.*?)(?=\[SCORE\])", retrievability_eval, re.DOTALL)
    retrievability_score_match = re.search(r"\[SCORE\]\s*([1-5])", retrievability_eval)
    evaluations["retrievability"] = retrievability_eval_match.group(1).strip() if retrievability_eval_match else retrievability_eval
    evaluations["retrievability_score"] = int(retrievability_score_match.group(1)) if retrievability_score_match else np.nan

    return evaluations


def evaluate_rag_metrics(eval_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Evaluates the generated answers using multiple metrics."""
    logger.info(f"LLM Evaluation model: {model_name}")

    with tqdm_joblib(desc="Evaluating Rows", total=eval_df.shape[0]) as progress_bar:
        evaluations = Parallel(n_jobs=-1)(
            delayed(evaluate_row)(row, model_name)
            for _, row in eval_df.iterrows()
        )

    evaluation_df = pd.DataFrame(evaluations)
    eval_df = eval_df.join(evaluation_df.set_index("question"), on=QUERY_COLUMN, rsuffix=f"_{model_name}")

    return eval_df


def evaluate_row(row, model_name: str) -> dict:
    """Evaluates a single row of data."""
    llm_client = OpenAI_Client()
    try:
        question = row[QUERY_COLUMN]
        context_extracted = row[RAG_RELEVANT_EXTRACTS_COLUMN]
        answer = row[ANSWER_COLUMN]

        if not context_extracted.strip():
            logger.warning(f"Empty context for question: {question}")
            return {
                "question": question,
                "model": model_name,
                "evaluation": "No context provided",
                "faithfulness_score": "No context provided",
                "correctness_score": "No context provided",
                "retrievability_score": "No context provided",
            }

        eval_results = evaluate_metrics(llm_client, question, answer, context_extracted)

        eval_entry = {"question": question, "model": model_name}
        eval_entry.update(eval_results)
        logger.info(f"Evaluations for question: {question} using model {model_name}: {eval_results}")
        return eval_entry

    except Exception as e:
        logger.error(f"Error evaluating metrics: {e}")
        return {
            "question": row[QUERY_COLUMN],
            "model": model_name,
            "evaluation": "Error during evaluation",
            "faithfulness_score": "Error during evaluation",
            "correctness_score": "Error during evaluation",
            "retrievability_score": "Error during evaluation",
        }

@app.command()
def main(
    qr_df_path: Path,
    report_output_path: Path = Path(__file__).parent / "report_output.xlsx"
):
    """Main function to evaluate the RAG pipeline."""
    try:
        if report_output_path.exists():
            existing_df = pd.read_excel(report_output_path)
            logger.info(f"Loaded existing evaluation file from {report_output_path}")
        else:
            existing_df = pd.DataFrame()
            logger.info("No existing evaluation file found. Starting fresh.")
    except Exception as e:
        logger.error(f"Failed to load existing evaluation file: {e}")
        existing_df = pd.DataFrame()

    eval_df = pd.read_excel(qr_df_path)
    logger.info(f"Loaded input dataset from {qr_df_path}")

    llm_client = OpenAI_Client()
    model_name = llm_client.model_name
    evaluated_df = evaluate_rag_metrics(eval_df, model_name)

    if not existing_df.empty:
        evaluated_df = evaluated_df.add_suffix(f"_{model_name}")
        merged_df = existing_df.merge(
            evaluated_df,
            left_on="question",
            right_on=f"question_{model_name}",
            how="outer",
            suffixes=('', f'_{model_name}')
        )
    else:
        merged_df = evaluated_df

    merged_df.to_excel(report_output_path, index=False)
    logger.info(f"Updated evaluation file saved to {report_output_path}")

if __name__ == "__main__":
    typer.run(main)