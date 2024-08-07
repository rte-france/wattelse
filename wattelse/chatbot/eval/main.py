import typer
from loguru import logger
import numpy as np
import pandas as pd

from pathlib import Path
from bert_score import score

from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.chatbot.eval.prompt import EVAL_LLM_PROMPT

QUERY_COLUMN = "query"
ANSWER_COLUMN = "answer"
DOC_LIST_COLUMN = "doc_list"

def main(
    qr_df_path: Path,
    eval_corpus_path: Path,
    output_path: Path = Path(__file__).parent / "eval_output.xlsx",
):
    """
    Function to evaluate the generation part of the RAG pipeline.
    Currently uses BERTScore and LLM as a judge as metrics.
    """

    # Initialize RAG backend and LLM client
    eval_group_id = "rag_eval"

    RAGBackEnd(eval_group_id).clear_collection()  # ensure RAG eval backend is empty
    RAG_EVAL_BACKEND = RAGBackEnd(eval_group_id)
    logger.info(f"RAG Backend LLM: {RAG_EVAL_BACKEND.llm.model_name}")

    EVAL_LLM_CLIENT = OpenAI_Client()
    logger.info(f"RAG evaluator LLM: {EVAL_LLM_CLIENT.model_name}")

    # Load data
    eval_df = pd.read_excel(qr_df_path)

    # Load eval docs in RAG backend
    for doc in eval_corpus_path.iterdir():
        if doc.is_file():
            with open(doc, "rb") as f:
                RAG_EVAL_BACKEND.add_file_to_collection(doc.name, f)
                logger.info(f"Added {doc.name} to collection")
        else:
            logger.warning(f"{doc} is not a file")

    # Get RAG predictions
    rag_answers = []
    rag_relevant_extracts = []
    for _, row in eval_df.iterrows():
        response = RAG_EVAL_BACKEND.query_rag(
            row[QUERY_COLUMN], selected_files=row[DOC_LIST_COLUMN]
        )
        answer = response[ANSWER_COLUMN]
        relevant_extracts = [
            extract["content"] for extract in response["relevant_extracts"]
        ]
        rag_answers.append(answer)
        rag_relevant_extracts.append(relevant_extracts)

    # Compute BERTscore
    refs = eval_df[ANSWER_COLUMN].tolist()

    P, R, F1 = score(rag_answers, refs, lang="fr")

    # Compute LLM as a judge score
    llm_scores = []
    for i, row in eval_df.iterrows():
        response = EVAL_LLM_CLIENT.generate(
            EVAL_LLM_PROMPT.format(
                query=row[QUERY_COLUMN],
                answer=row[ANSWER_COLUMN],
                candidate=rag_answers[i],
            )
        )
        llm_scores.append(response)

    # The following line parses the eval LLM output and should depend on the specific prompt used
    llm_scores = [int(score.split(":")[1].strip()) for score in llm_scores]

    # Log scores info
    logger.info(f"BERTScore P: {P.mean().item():.3f}")
    logger.info(f"BERTScore R: {R.mean().item():.3f}")
    logger.info(f"BERTScore F1: {F1.mean().item():.3f}")
    logger.info(f"LLM as a judge mean score: {np.mean(llm_scores):.3f}")

    # Update eval_df with RAG predictions and scores
    eval_df["rag_answer"] = rag_answers
    eval_df["rag_relevant_extracts"] = rag_relevant_extracts
    eval_df["bert_score_P"] = P.tolist()
    eval_df["bert_score_R"] = R.tolist()
    eval_df["bert_score_F1"] = F1.tolist()
    eval_df["llm_score"] = llm_scores

    # Save updated eval_df
    eval_df.to_excel(output_path, index=False)

    # Clear RAG backend
    RAG_EVAL_BACKEND.clear_collection()


if __name__ == "__main__":
    typer.run(main)
