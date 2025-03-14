import re
import typer
import pandas as pd
from loguru import logger
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from langchain.docstore.document import Document as LangchainDocument
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.indexer.document_parser import parse_file
from wattelse.indexer.document_splitter import split_file
from wattelse.evaluation_pipeline.synthethic_generation.prompt_QA_gen import (
    QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_SYNDICALE_TEST,
)

# TODO : The logic of the nuances is not flexible for QA Generation
# FIXME : Refactor the code too dependent on the previous versions + create a GeneConfig class

# Constants for default values
OUTPUT_PATH = Path("qa_output.xlsx")
REPORT_PATH = Path("report_output.xlsx")
DOCS_PER_QA = 1
CHUNKS_PER_DOC = 10
DEFAULT_GENERATIONS = 100
DEFAULT_MAX_TOKENS = 60000

# FIXME Can't set the chunk_size & chunk_overlap unless I modify document_splitter.py

# Define the Typer app
app = typer.Typer()


def split_documents(eval_corpus_path: Path) -> List[LangchainDocument]:
    docs_processed = []

    for file in eval_corpus_path.iterdir():
        if file.is_file():
            try:
                docs = parse_file(file)
            except Exception as e:
                logger.warning(f"Error parsing file {file.name}: {e}")
                continue

            try:
                splits = split_file(file.suffix.lower(), docs)
                docs_processed.extend(splits)
            except Exception as e:
                logger.warning(f"Error splitting file {file.name}: {e}")
                continue
    return docs_processed


def parse_generated_qa(output: str, labels: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse the generated QA content and extract questions and answers.

    Parameters:
    - output (str): The raw LLM response containing generated questions and answers.
    - labels (List[str]): The list of labels to categorize questions and answers.

    Returns:
    - Dict[str]
    """
    parsed_qa = {
        "questions": {},
        "answers": {},
    }

    for label in labels:
        # Handle both multiline and single line cases, stop at either Réponse or next Question
        question_match = re.search(
            rf"\*\*Question {label}\s*:\*\*\s*(.*?)(?=\s*\*\*(?:Réponse|Question)|\n\n|$)",
            output,
            re.DOTALL,
        )

        answer_match = re.search(
            rf"\*\*Réponse {label}\s*:\*\*\s*(.*?)(?=\s*\*\*Question|\n\n---|$)",
            output,
            re.DOTALL,
        )

        parsed_qa["questions"][label] = (
            question_match.group(1).strip() if question_match else ""
        )
        parsed_qa["answers"][label] = (
            answer_match.group(1).strip() if answer_match else ""
        )

    return parsed_qa


def prepare_chunk_batches(
    chunks_by_doc: Dict,
    n_generations: int = DEFAULT_GENERATIONS,
    docs_per_qa: int = DOCS_PER_QA,
    chunks_per_doc: int = CHUNKS_PER_DOC,
) -> List[List[LangchainDocument]]:
    """
    Prepare document chunk batches with improved utilization of available chunks.
    Allows partial chunk usage when a document has fewer chunks than chunks_per_doc.
    """
    all_batches = []
    chunk_usage = {
        doc_name: [False] * len(chunks) for doc_name, chunks in chunks_by_doc.items()
    }

    for _ in range(n_generations):
        current_batch = []
        used_docs = set()

        # Try to form a complete batch
        while len(current_batch) < docs_per_qa * chunks_per_doc and len(
            used_docs
        ) < len(chunks_by_doc):
            # Get sorted docs by available chunks percentage
            doc_chunks_remaining = {
                doc: sum(1 for used in usage if not used) / len(usage)
                for doc, usage in chunk_usage.items()
                if sum(1 for used in usage if not used)
                > 0  # Only consider docs with available chunks
            }

            if not doc_chunks_remaining:
                break

            high_yield_docs = sorted(
                doc_chunks_remaining, key=doc_chunks_remaining.get, reverse=True
            )

            selected_doc = None
            # Find an unused doc with any available chunks
            for doc_name in high_yield_docs:
                if doc_name not in used_docs:
                    selected_doc = doc_name
                    used_docs.add(doc_name)
                    break

            if selected_doc is None:
                break

            # Select chunks from the chosen document
            doc_chunks = chunks_by_doc[selected_doc]
            available_chunks = [
                (idx, chunk)
                for idx, (chunk, used) in enumerate(
                    zip(doc_chunks, chunk_usage[selected_doc])
                )
                if not used
            ]

            # Take as many chunks as possible up to chunks_per_doc
            chunks_to_take = min(len(available_chunks), chunks_per_doc)
            if chunks_to_take > 0:
                chunks_to_use = available_chunks[:chunks_to_take]
                for idx, chunk in chunks_to_use:
                    chunk_usage[selected_doc][idx] = True
                    current_batch.append(chunk)
                    logger.info(f"Using chunk from {Path(selected_doc).name}: [{idx}]")

        # Add batch if it has at least one document's worth of chunks
        if len(current_batch) >= chunks_per_doc:
            # Pad the batch if necessary to maintain consistent size
            while len(current_batch) < docs_per_qa * chunks_per_doc:
                # Duplicate some chunks to maintain the expected batch size
                current_batch.append(current_batch[0])
            all_batches.append(current_batch)
        else:
            break

    return all_batches


def process_single_generation(batch: List[LangchainDocument]) -> Dict:
    """Process a single QA generation with pre-selected document chunks."""
    if not batch:
        return None

    llm_client = OpenAI_Client()
    llm_client.max_tokens = DEFAULT_MAX_TOKENS

    try:
        combined_context = "\n\n".join(chunk.page_content for chunk in batch)
        output_QA_couple = llm_client.generate(
            QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_SYNDICALE_TEST.format(
                context=combined_context
            ),
            temperature=0,
        )

        labels = ["simple", "de raisonnement", "multi-contexte"]
        parsed_qa = parse_generated_qa(output_QA_couple, labels)

        # assert all(len(ans) < 2000 for ans in parsed_qa["answers"].values()), "Answer is too long"

        source_docs = [Path(chunk.metadata["source"]).name for chunk in batch]
        return {
            "context": combined_context,
            "questions": {
                "simple": parsed_qa["questions"]["simple"],
                "reasoning": parsed_qa["questions"]["de raisonnement"],
                "multi_context": parsed_qa["questions"]["multi-contexte"],
            },
            "answers": {
                "simple": parsed_qa["answers"]["simple"],
                "reasoning": parsed_qa["answers"]["de raisonnement"],
                "multi_context": parsed_qa["answers"]["multi-contexte"],
            },
            "source_docs": source_docs,
        }
    except Exception as e:
        logger.error(f"Error generating QA pair: {e}")
        return None


def generate_qa_pairs(
    eval_corpus_path: Path,
    n_generations: int = DEFAULT_GENERATIONS,
    output_path: Path = OUTPUT_PATH,
    docs_per_qa: int = DOCS_PER_QA,
    chunks_per_doc: int = CHUNKS_PER_DOC,
) -> List[Dict]:

    # Process and split documents
    docs_processed = split_documents(eval_corpus_path)

    # Group chunks by document
    chunks_by_doc = defaultdict(list)
    for chunk in docs_processed:
        source = chunk.metadata["source"]
        chunks_by_doc[source].append(chunk)

    # Prepare all batches upfront
    logger.info("Preparing document chunks for parallel processing...")
    chunk_batches = prepare_chunk_batches(
        chunks_by_doc, n_generations, docs_per_qa, chunks_per_doc
    )

    actual_generations = len(chunk_batches)
    logger.info(f"Prepared {actual_generations} complete batches for processing")

    if actual_generations == 0:
        logger.warning("No complete batches could be formed with the given constraints")
        return []

    # Process batches in parallel
    logger.info("Starting parallel QA generation...")
    with tqdm_joblib(tqdm(desc="Generating QA pairs", total=actual_generations)):
        outputs = Parallel(n_jobs=-1)(
            delayed(process_single_generation)(batch) for batch in chunk_batches
        )

    # Filter out None values
    outputs = [output for output in outputs if output is not None]

    # Save results
    qa_df = pd.DataFrame(outputs)
    qa_df.to_excel(output_path, index=False)
    logger.info(f"Generated QA dataset saved to {output_path}")

    # Generate utilization summary
    logger.info("Document Utilization Summary:")
    chunks_used = defaultdict(int)
    for batch in chunk_batches:
        for chunk in batch:
            chunks_used[chunk.metadata["source"]] += 1

    for doc_name, chunks in chunks_by_doc.items():
        used = chunks_used[doc_name]
        total = len(chunks)
        logger.success(
            f"{Path(doc_name).name}: {used}/{total} chunks used ({(used/total)*100:.2f}% utilization)"
        )

    return outputs


@app.command()
def main(
    eval_corpus_path: Path = Path,
    n_generations: int = DEFAULT_GENERATIONS,
    output_path: Path = OUTPUT_PATH,
):
    """
    Usage :
    python generate_qa.py --eval-corpus-path data/eval_demo --n-generations 10 --output-path output_qa_DEMO.xlsx --report-output-path report_output_DEMO.xlsx

    - eval_corpus_path: Path = Path,  # Default input path
    - n_generations: int = 100,  # Number of generations by default, It caps when it reaches the maximum chunks possible from the available documents.
    - output_path: Path = Path(__file__).parent / "qa_output.xlsx",  # Default output path

    Function to generate the synthethic data part of the RAG pipeline.
    Currently supports multiple complexity.
    """
    qa_pairs = generate_qa_pairs(eval_corpus_path, n_generations, output_path)

    if qa_pairs:
        output_data = []
        for output in qa_pairs:
            for complexity in ["simple", "reasoning", "multi_context"]:
                question = output["questions"][complexity]
                output_data.append(
                    {
                        "context": output["context"],
                        "question": question,
                        "answer": output["answers"][complexity],
                        "complexity": complexity,
                        "source_doc": ", ".join(output["source_docs"]),
                    }
                )

        # Save the final evaluated QA pairs to an Excel file
        df_output = pd.DataFrame(output_data)
        df_output.to_excel(output_path, index=False)
        logger.success(f"Synthetic QA saved to {output_path}")


# Run the app
if __name__ == "__main__":
    app()
