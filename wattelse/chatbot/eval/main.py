from loguru import logger
import numpy as np
import pandas as pd

from pathlib import Path
from bert_score import score

from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.api.openai.client_openai_api import OpenAI_Client


EVAL_DATASET_DIR_PATH = Path(__file__).parent / "eval_dataset"

EVAL_GROUP_ID = "rag_eval"

RAGBackEnd(EVAL_GROUP_ID).clear_collection()  # ensure RAG eval backend is empty
RAG_EVAL_BACKEND = RAGBackEnd(EVAL_GROUP_ID)
logger.info(f"RAG Backend LLM: {RAG_EVAL_BACKEND.llm}")

EVAL_LLM_CLIENT = OpenAI_Client()
logger.info(f"RAG evaluator LLM: {EVAL_LLM_CLIENT.model_name}")

EVAL_LLM_PROMPT = (
    "You are an evaluator. You will be provided with a a query, "
    "the groundtruth answer and a candidate response. You must "
    "evaluate the candidate response based on the groundtruth "
    "answer and the query. You must provide one of the following "
    "scores:\n"
    "- 0: the candidate response is incorrect and contains wrong information\n"
    "- 1: the candidate response is not incorrect but miss important parts "
    "of the groundtruth answer\n"
    "- 2: the candidate response is correct but miss some details that "
    "do not impact the veracity of the information\n"
    "- 3: the candidate response is correct and provides all the "
    "information from the groundtruth answer\n"
    'You must answer with the score only, using the format "Score:"\n\n'
    "Query: {query}\n"
    "Groundtruth answer: {answer}\n"
    "Candidate response: {candidate}\n"
)

# Load data
eval_df = pd.read_excel(EVAL_DATASET_DIR_PATH / "rag_eval.xlsx")

# Load eval docs in RAG backend
for doc in (EVAL_DATASET_DIR_PATH / "corpus").iterdir():
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
        row["query"], selected_files=[row["doc_name"]]
    )
    answer = response["answer"]
    relevant_extracts = [
        extract["content"] for extract in response["relevant_extracts"]
    ]
    rag_answers.append(answer)
    rag_relevant_extracts.append(relevant_extracts)


# Compute BERTscore
refs = eval_df["answer"].tolist()

P, R, F1 = score(rag_answers, refs, lang="fr")

# Compute LLM as a judge score
llm_scores = []
for i, row in eval_df.iterrows():
    response = EVAL_LLM_CLIENT.generate(
        EVAL_LLM_PROMPT.format(
            query=row["query"],
            answer=row["answer"],
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
eval_df.to_excel(EVAL_DATASET_DIR_PATH / "evaluated.xlsx")