import re
import typer
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.evaluation_pipeline.eval_config import EvalConfig

CONFIG_EVAL = Path("config.cfg")
REPORT_PATH = Path("report_output.xlsx")

# Column definitions
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

def parse_eval_response(eval_text, metric_key, regex_patterns):
    """Parse evaluation responses using configured regex patterns."""
    eval_match = re.search(regex_patterns["evaluation"], eval_text, re.DOTALL)
    score_match = re.search(regex_patterns["judgment"], eval_text)
    
    return {
        f"{metric_key}": eval_match.group(1).strip() if eval_match else eval_text,
        f"{metric_key}_score": int(score_match.group(1)) if score_match else np.nan
    }
def evaluate_metrics(llm_client, question, answer, context_extracted, config: EvalConfig) -> dict:
    """
    Evaluates the answer based on configured metrics using the LLM.

    Args:
        llm_client (OpenAI_Client): The client to interact with the OpenAI LLM.
        question (str): The question to evaluate.
        answer (str): The answer to evaluate.
        context_extracted (str): The context used to generate the answer.
        config (EvalConfig): Configuration object for model-specific settings.

    Returns:
        dict: A dictionary containing the evaluations and scores for enabled metrics.
    """
    # Fetch model-specific configurations
    model_name = getattr(llm_client, 'model_name', config.default_model)
    regex_patterns = config.get_regex_patterns(model_name)

    kwargs = {"max_tokens": 4096}
    
    evaluations = {}
    
    # Only evaluate enabled and available metrics
    for metric in config.active_metrics:
        try:
            prompt = config.get_prompt(metric, model_name)
            
            # Format prompt based on metric type
            if metric == "faithfulness":
                eval_text = llm_client.generate(
                    prompt.format(
                        retrieved_contexts=context_extracted,
                        answer=answer
                    ),
                    **kwargs
                )
            elif metric == "correctness":
                eval_text = llm_client.generate(
                    prompt.format(
                        question=question,
                        answer=answer
                    ),
                    **kwargs
                )
            elif metric == "retrievability":
                eval_text = llm_client.generate(
                    prompt.format(
                        question=question,
                        retrieved_contexts=context_extracted
                    ),
                    **kwargs
                )
            else:
                logger.warning(f"Unknown metric type: {metric}")
                continue
                
            logger.debug(f"{metric} LLM response: {eval_text}")
            evaluations.update(parse_eval_response(eval_text, metric, regex_patterns))
            
        except Exception as e:
            logger.error(f"Error evaluating {metric}: {e}")
            evaluations.update({
                f"{metric}": f"Error: {str(e)}",
                f"{metric}_score": np.nan
            })
    
    return evaluations

def evaluate_row(row: pd.Series, config: EvalConfig) -> dict:
    """Function to evaluate a single row of data (question, answer, and context)."""
    llm_client = OpenAI_Client()  # Initialize the LLM client inside the worker function
    try:
        question = row[QUERY_COLUMN]
        context_extracted = row[RAG_RELEVANT_EXTRACTS_COLUMN]
        answer = row[ANSWER_COLUMN]

        if not context_extracted.strip():
            logger.warning(f"Empty context for question: {question}")
            return {
                "question": question,
                "evaluation": "No context provided",
            }

        eval_results = evaluate_metrics(llm_client, question, answer, context_extracted, config)

        eval_entry = {"question": question}
        eval_entry.update(eval_results)
        logger.info(f"Evaluations for question: {question}: {eval_results}")
        return eval_entry

    except Exception as e:
        logger.error(f"Error evaluating metrics: {e}")
        return {
            "question": row[QUERY_COLUMN],
            "evaluation": "Error during evaluation",
        }

def evaluate_rag_metrics(eval_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """
    Evaluates the generated answers from the RAG pipeline using multiple metrics.

    Args:
        eval_df (pandas.DataFrame): The DataFrame containing the evaluation corpus and answers.
        config (EvalConfig): Configuration object for evaluation settings.

    Returns:
        pandas.DataFrame: The DataFrame with added evaluation columns.
    """
    logger.info(f"LLM Evaluation model: {OpenAI_Client().model_name}")

    # Wrap the Parallel execution with tqdm_joblib to show progress
    with tqdm_joblib(desc="Evaluating Rows", total=eval_df.shape[0]) as progress_bar:
        evaluations = Parallel(n_jobs=-1)(
            delayed(evaluate_row)(row, config)
            for _, row in eval_df.iterrows()
        )

    # Combine evaluations into a DataFrame
    evaluation_df = pd.DataFrame(evaluations)

    # Join the evaluations to the original DataFrame
    eval_df = eval_df.join(evaluation_df.set_index("question"), on=QUERY_COLUMN, rsuffix="_eval")

    return eval_df

@app.command()
def main(
    qr_df_path: Path,
    config_path: Path = CONFIG_EVAL,
    report_output_path: Path = REPORT_PATH
):
    """Main function to evaluate the RAG pipeline."""
    config = EvalConfig(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    eval_df = pd.read_excel(qr_df_path)
    logger.info(f"Loaded input dataset from {qr_df_path}")

    evaluated_df = evaluate_rag_metrics(eval_df, config)
    
    evaluated_df.to_excel(report_output_path, index=False)
    logger.info(f"Evaluation results saved to {report_output_path}")

if __name__ == "__main__":
    typer.run(main)