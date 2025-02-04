import re
import time
import typer
import pandas as pd
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from wattelse.api.embedding.client_embedding_api import EmbeddingAPI

CONFIG_RAG = "azure_20241216"
QUERY_COLUMN = "question"
DOC_LIST_COLUMN = "source_doc"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"
RAG_QUERY_TIME_COLUMN = "rag_query_time_seconds"
RAG_RETRIEVER_TIME_COLUMN = "rag_retriever_time_seconds"
SPECIAL_CHARACTER_FILTER = (
    r"[\t\n\r\x07\x08\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009]"
)

# Configuration sheet names
RAG_CONFIG_SHEET_NAME = "prompts_config"
RETRIEVER_CONFIG_SHEET_NAME = "retriever_config"
GENERATOR_CONFIG_SHEET_NAME = "generator_config"
COLLECTION_CONFIG_SHEET_NAME = "collection_config"
QUERY_LENGTH_CHARS = "question_length_chars"
QUERY_LENGTH_WORDS = "question_length_words"

app = typer.Typer()

@app.command()
def main(
    qr_df_path: Path,
    eval_corpus_path: Path,
    predictions_output_path: Path = Path(__file__).parent / "predictions_output.xlsx"
):
    """
    python rag_predictions.py <qr_df_path> <eval_corpus_path> --report-output-path <output_path>

    Main function to generate predictions from the RAG pipeline.
    This function retrieves answers using RAG without performing any evaluation.

    Args:
        qr_df_path (Path): Path to the query and answer dataset (in Excel format).
        eval_corpus_path (Path): Path to the evaluation corpus folder.
        predictions_output_path (Path, optional): Path to save the predictions (Excel file).
    """
    # Initialize RAG backend

    eval_group_id = "rag_eval"
    RAGBackEnd(eval_group_id, CONFIG_RAG).clear_collection()
    RAG_EVAL_BACKEND = RAGBackEnd(eval_group_id, CONFIG_RAG)
    logger.info(f"RAG Backend initialized with LLM: {RAG_EVAL_BACKEND.llm.model_name}")

    EMBEDDING_API = EmbeddingAPI()
    # Load data
    eval_df = pd.read_excel(qr_df_path)

    # Transform source_doc to a list
    eval_df[DOC_LIST_COLUMN] = eval_df[DOC_LIST_COLUMN].apply(
        lambda x: [doc.strip() for doc in x.split(",")] if pd.notnull(x) else []
    )

    # Flatten and deduplicate all document names
    all_doc_list = set(
        [item for sublist in eval_df[DOC_LIST_COLUMN].to_list() for item in sublist]
    )

    # Validate documents exist
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

    # Get Prompts configurations
    prompt_configs = {
        "System Prompt": RAG_EVAL_BACKEND.system_prompt,
        "User Prompt": RAG_EVAL_BACKEND.user_prompt,
        "System Prompt Query Contextualization": RAG_EVAL_BACKEND.system_prompt_query_contextualization,
        "User Prompt Query Contextualization": RAG_EVAL_BACKEND.user_prompt_query_contextualization,
    }

    # Get detailed retriever configuration
    retriever_configs = {
        "Embedding Model": EMBEDDING_API.get_api_model_name(),
        "Retrieval Method": RAG_EVAL_BACKEND.retrieval_method,
        "Top N Extracts": RAG_EVAL_BACKEND.top_n_extracts,
        "Similarity Threshold": RAG_EVAL_BACKEND.similarity_threshold,
    }

    # Get LLM/Generator configuration
    generator_configs = {
        "LLM Model": RAG_EVAL_BACKEND.get_llm_model_name(),
        "Temperature": RAG_EVAL_BACKEND.temperature,
        "Remember Recent Messages": RAG_EVAL_BACKEND.remember_recent_messages,
        "Multi Query Mode": RAG_EVAL_BACKEND.multi_query_mode,
    }

    # Get Collection configuration
    collection_configs = {
        "Collection Name": RAG_EVAL_BACKEND.document_collection.collection_name,
        "Total Chunks": len(RAG_EVAL_BACKEND.get_text_list(None)),
        "Total Selected Documents": len(all_doc_list),
        "Total Documents Available": len(RAG_EVAL_BACKEND.get_available_docs()),
        "Selected Documents": ", ".join(sorted(all_doc_list)) if all_doc_list else "None",
        "Documents In Use": ", ".join(sorted(all_doc_list)),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Log all configurations
    for config_name, config_dict in [
        ("Prompts", prompt_configs),
        ("Retriever", retriever_configs),
        ("Generator", generator_configs),
        ("Collection", collection_configs)
    ]:
        logger.info(f"\n{config_name} Configuration:")
        for key, value in config_dict.items():
            logger.info(f"{key}: {value}")

    # Get RAG predictions
    rag_answers = []
    rag_relevant_extracts = []
    rag_query_times = []
    rag_retriever_times = []
    
    for _, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Getting RAG Predictions"):
        logger.info(f"Processing row: question={row[QUERY_COLUMN]}")

        # Measure retriever time
        retriever_start_time = time.time()
        retriever = RAG_EVAL_BACKEND.document_collection.collection.as_retriever()
        _ = retriever.get_relevant_documents(row[QUERY_COLUMN])
        retriever_time = time.time() - retriever_start_time

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
        logger.info(f"Retriever time: {retriever_time:.2f} seconds")

        rag_answers.append(answer)
        rag_relevant_extracts.append(relevant_extracts)
        rag_query_times.append(query_time)
        rag_retriever_times.append(retriever_time)

    # Save predictions to DataFrame
    eval_df["rag_answer"] = rag_answers
    eval_df[RAG_RELEVANT_EXTRACTS_COLUMN] = [
        "\n\n".join([f"Extrait {i + 1}: {extract}" for i, extract in enumerate(sublist)])
        for sublist in rag_relevant_extracts
    ]
    eval_df[RAG_QUERY_TIME_COLUMN] = rag_query_times
    eval_df[RAG_RETRIEVER_TIME_COLUMN] = rag_retriever_times

    eval_df[QUERY_LENGTH_CHARS] = eval_df[QUERY_COLUMN].str.len()
    eval_df[QUERY_LENGTH_WORDS] = eval_df[QUERY_COLUMN].str.split().str.len()

    # Save predictions to DataFrame
    eval_df["rag_answer"] = rag_answers

    # Log timing statistics
    logger.info(f"Average query time: {pd.Series(rag_query_times).mean():.2f} seconds")
    logger.info(f"Maximum query time: {pd.Series(rag_query_times).max():.2f} seconds")
    logger.info(f"Minimum query time: {pd.Series(rag_query_times).min():.2f} seconds")
    logger.info(f"Average retriever time: {pd.Series(rag_retriever_times).mean():.2f} seconds")
    logger.info(f"Maximum retriever time: {pd.Series(rag_retriever_times).max():.2f} seconds")
    logger.info(f"Minimum retriever time: {pd.Series(rag_retriever_times).min():.2f} seconds")

    # Save all data to Excel with multiple sheets
    with pd.ExcelWriter(predictions_output_path, engine='openpyxl') as writer:
        eval_df.to_excel(writer, sheet_name='predictions', index=False)
        rag_df = pd.DataFrame(list(prompt_configs.items()), columns=['Parameter', 'Value'])
        retriever_df = pd.DataFrame(list(retriever_configs.items()), columns=['Parameter', 'Value'])
        generator_df = pd.DataFrame(list(generator_configs.items()), columns=['Parameter', 'Value'])
        collection_df = pd.DataFrame(list(collection_configs.items()), columns=['Parameter', 'Value'])
        
        rag_df.to_excel(writer, sheet_name=RAG_CONFIG_SHEET_NAME, index=False)
        retriever_df.to_excel(writer, sheet_name=RETRIEVER_CONFIG_SHEET_NAME, index=False)
        generator_df.to_excel(writer, sheet_name=GENERATOR_CONFIG_SHEET_NAME, index=False)
        collection_df.to_excel(writer, sheet_name=COLLECTION_CONFIG_SHEET_NAME, index=False)

    logger.info(f"Predictions and configurations saved to {predictions_output_path}")

    # Clear RAG backend
    RAG_EVAL_BACKEND.clear_collection()

if __name__ == "__main__":
    typer.run(main)