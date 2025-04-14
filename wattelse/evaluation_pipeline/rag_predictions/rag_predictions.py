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
from wattelse.rag_backend.rag_backend import RAGBackend
from wattelse.evaluation_pipeline import (
    BASE_DIR,
    BASE_DATA_DIR,
    BASE_DOCS_DIR,
    BASE_OUTPUT_DIR,
)

CONFIG_RAG = "mistral_20250123"
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


def resolve_path(path, base_dir):
    """
    Resolves a path that might be relative or just a filename to a full path.
    Handles several cases:
    1. Absolute path: return as is
    2. Relative path that exists: return as is
    3. Path starting with 'data/': resolve against experiment root
    4. Path starting with 'docs/': resolve against experiment root
    5. Simple filename: prepend the appropriate base directory
    """
    path = Path(path)

    # If it's an absolute path or exists as is, return it
    if path.is_absolute() or path.exists():
        return path

    # Handle paths starting with 'data/' or 'docs/'
    path_str = str(path)
    if path_str.startswith("data/"):
        return BASE_DIR / path_str
    if path_str.startswith("docs/"):
        return BASE_DIR / path_str

    # Otherwise, join with base directory
    return base_dir / path


@app.command()
def main(
    qr_df_path: Path = typer.Argument(
        ..., help="Path or filename of the query and answer dataset (Excel format)"
    ),
    eval_corpus_path: Path = typer.Argument(
        ..., help="Path or directory name of the evaluation corpus folder"
    ),
    predictions_output_path: Path = typer.Option(
        None,
        "--predictions-output-path",
        "-o",
        help="Path or filename for the predictions output (Excel file)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing output file if it exists",
    ),
):
    """
    Main function to generate predictions from the RAG pipeline.
    This function retrieves answers using RAG without performing any evaluation.

    Example usage with simple filenames:
    python rag_predictions.py Corpus-général.xlsx eval_align -o PHIs.xlsx

    Example usage with relative paths:
    python rag_predictions.py data/Corpus-général.xlsx docs/eval_align -o PHIs.xlsx

    Example usage with full paths:
    python rag_predictions.py /DSIA/nlp/experiments/data/Corpus-général.xlsx /DSIA/nlp/experiments/docs/eval_align -o /DSIA/nlp/experiments/data_predictions/PHIs.xlsx
    """
    # Resolve paths based on base directories if they are not absolute
    qr_df_path = resolve_path(qr_df_path, BASE_DATA_DIR)
    eval_corpus_path = resolve_path(eval_corpus_path, BASE_DOCS_DIR)

    # If output path not provided, create one based on input filename
    if predictions_output_path is None:
        input_filename = qr_df_path.stem
        predictions_output_path = BASE_OUTPUT_DIR / f"{input_filename}_predictions.xlsx"
    else:
        # Resolve output path
        predictions_output_path = resolve_path(predictions_output_path, BASE_OUTPUT_DIR)

    # Check if output file exists and handle accordingly
    if predictions_output_path.exists() and not overwrite:
        counter = 1
        while True:
            new_path = predictions_output_path.with_name(
                f"{predictions_output_path.stem}_{counter}{predictions_output_path.suffix}"
            )
            if not new_path.exists():
                logger.warning(f"Output file {predictions_output_path} already exists.")
                predictions_output_path = new_path
                logger.info(f"Using alternative path: {predictions_output_path}")
                break
            counter += 1

    # Ensure output directory exists
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input data: {qr_df_path}")
    logger.info(f"Corpus directory: {eval_corpus_path}")
    logger.info(f"Output file: {predictions_output_path}")

    # Initialize RAG backend
    eval_group_id = "rag_eval"

    # First, create a RAG backend instance
    RAG_EVAL_BACKEND = RAGBackend(eval_group_id, CONFIG_RAG)

    # Then clear the collection
    try:
        RAG_EVAL_BACKEND.clear_collection()
        logger.info("Cleared existing collection")
    except Exception as e:
        logger.warning(f"Error clearing collection: {e}")

    # After clearing, recreate the RAG backend to initialize a fresh collection
    # This is necessary because clear_collection() deletes the collection completely
    RAG_EVAL_BACKEND = RAGBackend(eval_group_id, CONFIG_RAG)

    logger.info(f"RAG Backend initialized with LLM: {RAG_EVAL_BACKEND.llm.model_name}")

    # Get embedding API model name directly from the RAG Backend's config
    embedding_model_name = RAG_EVAL_BACKEND.config.retriever.embedding_model_name

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
    documents_added = False
    for doc in tqdm(
        eval_corpus_path.iterdir(), desc="Loading documents into RAG Backend"
    ):
        if doc.name in all_doc_list:
            try:
                with open(doc, "rb") as f:
                    RAG_EVAL_BACKEND.add_file_to_collection(doc.name, f)
                    logger.info(f"Added {doc.name} to collection")
                    documents_added = True
            except Exception as e:
                logger.error(f"Error adding {doc.name} to collection: {e}")
                # If we have an error, we might need to recreate the backend
                if not documents_added:
                    logger.warning("Document loading failed, recreating RAG backend")
                    # Recreate the RAG backend instance to ensure fresh collection
                    RAG_EVAL_BACKEND = RAGBackend(eval_group_id, CONFIG_RAG)
                    # Try again with the same document
                    try:
                        with open(doc, "rb") as f_retry:
                            RAG_EVAL_BACKEND.add_file_to_collection(doc.name, f_retry)
                            logger.info(
                                f"Successfully added {doc.name} after recreating backend"
                            )
                            documents_added = True
                    except Exception as e_retry:
                        logger.error(f"Failed on retry: {e_retry}")
                        raise ValueError(
                            "Could not add documents to collection after recreating the backend"
                        )

    # Make sure at least some documents were added
    if not documents_added:
        raise ValueError("No documents were successfully added to the collection")

    # Get Prompts configurations
    prompt_configs = {
        "System Prompt": RAG_EVAL_BACKEND.config.generator.system_prompt,
        "User Prompt": RAG_EVAL_BACKEND.config.generator.user_prompt,
        "System Prompt Query Contextualization": RAG_EVAL_BACKEND.config.generator.system_prompt_query_contextualization,
        "User Prompt Query Contextualization": RAG_EVAL_BACKEND.config.generator.user_prompt_query_contextualization,
    }

    # Get detailed retriever configuration
    retriever_configs = {
        "Embedding Model": embedding_model_name,
        "Retrieval Method": RAG_EVAL_BACKEND.config.retriever.retrieval_method,
        "Top N Extracts": RAG_EVAL_BACKEND.config.retriever.top_n_extracts,
        "Similarity Threshold": RAG_EVAL_BACKEND.config.retriever.similarity_threshold,
    }

    # Get LLM/Generator configuration
    generator_configs = {
        "LLM Model": RAG_EVAL_BACKEND.get_llm_model_name(),
        "Temperature": RAG_EVAL_BACKEND.config.generator.temperature,
        "Remember Recent Messages": RAG_EVAL_BACKEND.config.generator.remember_recent_messages,
        "Multi Query Mode": RAG_EVAL_BACKEND.config.retriever.multi_query_mode,
    }

    # Get Collection configuration
    collection_configs = {
        "Collection Name": RAG_EVAL_BACKEND.document_collection.collection_name,
        "Total Chunks": len(RAG_EVAL_BACKEND.get_text_list(None)),
        "Total Selected Documents": len(all_doc_list),
        "Total Documents Available": len(RAG_EVAL_BACKEND.get_available_docs()),
        "Selected Documents": (
            ", ".join(sorted(all_doc_list)) if all_doc_list else "None"
        ),
        "Documents In Use": ", ".join(sorted(all_doc_list)),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Log all configurations
    for config_name, config_dict in [
        ("Prompts", prompt_configs),
        ("Retriever", retriever_configs),
        ("Generator", generator_configs),
        ("Collection", collection_configs),
    ]:
        logger.info(f"\n{config_name} Configuration:")
        for key, value in config_dict.items():
            logger.info(f"{key}: {value}")

    # Get RAG predictions
    rag_answers = []
    rag_relevant_extracts = []
    rag_query_times = []
    rag_retriever_times = []

    for _, row in tqdm(
        eval_df.iterrows(), total=eval_df.shape[0], desc="Getting RAG Predictions"
    ):
        logger.info(f"Processing row: question={row[QUERY_COLUMN]}")

        # Measure retriever time
        retriever_start_time = time.time()
        try:
            retriever = RAG_EVAL_BACKEND.document_collection.collection.as_retriever()
            _ = retriever.get_relevant_documents(row[QUERY_COLUMN])
            retriever_time = time.time() - retriever_start_time
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            retriever_time = 0

        # Measure query time
        start_time = time.time()
        try:
            response = RAG_EVAL_BACKEND.query_rag(
                row[QUERY_COLUMN], selected_files=row[DOC_LIST_COLUMN]
            )
            query_time = time.time() - start_time

            answer = response.get("answer", "")
            relevant_extracts = [
                re.sub(SPECIAL_CHARACTER_FILTER, " ", extract["content"])
                for extract in response.get(
                    "relevant_extracts", []
                )  # TODO: Modify this to rag_relevant_extracts
            ]
        except Exception as e:
            logger.error(f"Error during query: {e}")
            query_time = 0
            answer = f"Error: {str(e)}"
            relevant_extracts = []

        logger.info(f"RAG Answer for question '{row[QUERY_COLUMN]}': {answer}")
        logger.info(f"RAG Relevant Extracts: {relevant_extracts}")
        logger.info(f"Query time: {query_time:.2f} seconds")
        logger.info(f"Retriever time: {retriever_time:.2f} seconds")

        rag_answers.append(answer)
        rag_relevant_extracts.append(relevant_extracts)
        rag_query_times.append(query_time)
        rag_retriever_times.append(retriever_time)

    # Save predictions to DataFrame
    eval_df["answer"] = rag_answers
    eval_df[RAG_RELEVANT_EXTRACTS_COLUMN] = [
        "\n\n".join(
            [f"Extrait {i + 1}: {extract}" for i, extract in enumerate(sublist)]
        )
        for sublist in rag_relevant_extracts
    ]
    eval_df[RAG_QUERY_TIME_COLUMN] = rag_query_times
    eval_df[RAG_RETRIEVER_TIME_COLUMN] = rag_retriever_times

    eval_df[QUERY_LENGTH_CHARS] = eval_df[QUERY_COLUMN].str.len()
    eval_df[QUERY_LENGTH_WORDS] = eval_df[QUERY_COLUMN].str.split().str.len()

    # Log timing statistics
    logger.info(f"Average query time: {pd.Series(rag_query_times).mean():.2f} seconds")
    logger.info(f"Maximum query time: {pd.Series(rag_query_times).max():.2f} seconds")
    logger.info(f"Minimum query time: {pd.Series(rag_query_times).min():.2f} seconds")
    logger.info(
        f"Average retriever time: {pd.Series(rag_retriever_times).mean():.2f} seconds"
    )
    logger.info(
        f"Maximum retriever time: {pd.Series(rag_retriever_times).max():.2f} seconds"
    )
    logger.info(
        f"Minimum retriever time: {pd.Series(rag_retriever_times).min():.2f} seconds"
    )

    # Save all data to Excel with multiple sheets
    with pd.ExcelWriter(predictions_output_path, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="predictions", index=False)
        rag_df = pd.DataFrame(
            list(prompt_configs.items()), columns=["Parameter", "Value"]
        )
        retriever_df = pd.DataFrame(
            list(retriever_configs.items()), columns=["Parameter", "Value"]
        )
        generator_df = pd.DataFrame(
            list(generator_configs.items()), columns=["Parameter", "Value"]
        )
        collection_df = pd.DataFrame(
            list(collection_configs.items()), columns=["Parameter", "Value"]
        )

        rag_df.to_excel(writer, sheet_name=RAG_CONFIG_SHEET_NAME, index=False)
        retriever_df.to_excel(
            writer, sheet_name=RETRIEVER_CONFIG_SHEET_NAME, index=False
        )
        generator_df.to_excel(
            writer, sheet_name=GENERATOR_CONFIG_SHEET_NAME, index=False
        )
        collection_df.to_excel(
            writer, sheet_name=COLLECTION_CONFIG_SHEET_NAME, index=False
        )

    logger.info(f"Predictions and configurations saved to {predictions_output_path}")

    # Try to properly clean up, but don't crash if it fails
    try:
        # Don't clear the collection here as it might cause issues
        # if we need to reuse the same backend later
        pass
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    typer.run(main)
