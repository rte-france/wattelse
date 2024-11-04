
import typer
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict
from tqdm import tqdm
from docx import Document
from collections import defaultdict
from pathlib import Path
import PyPDF2
from wattelse.api.openai.client_openai_api import OpenAI_Client 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from collections import defaultdict
import re



# TO DO : Use the classes rather than creating the functions.
# from wattelse.indexer.document_parser import parse_file
# from wattelse.indexer.document_splitter import split_file

from wattelse.chatbot.eval.prompt import (
    QA_GENERATION_PROMPT,
    QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT,
    QUESTION_REALISM_CRITIQUE_PROMPT,
    QUESTION_STANDALONE_CRITIQUE_PROMPT
)

# Example :
# python generate_qa.py --eval-corpus-path data/eval_one --n-generations 5 --output-path output_qa_102.xlsx --report-output-path report_output_102.xlsx


# Define the Typer app
app = typer.Typer()

# Function to call the LLM
def call_llm(llm_client, prompt: str):
    response = llm_client.generate(prompt, temperature=0)
    return response

# Function to split documents Update with Langchain and .DOCX and others...
def split_documents(eval_corpus_path: Path):
    docs_processed = []
    for doc in eval_corpus_path.iterdir():
        content = ""
        if doc.suffix == ".pdf":
            # Handle PDF files
            try:
                with open(doc, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text() is not None])
            except Exception as e:
                print(f"Error reading {doc.name}: {e}")
                continue
        elif doc.suffix == ".docx":
            # Handle DOCX files
            try:
                docx_file = Document(doc)
                content = "\n".join([paragraph.text for paragraph in docx_file.paragraphs if paragraph.text.strip() != ""])
            except Exception as e:
                print(f"Error reading {doc.name}: {e}")
                continue
        else:
            # Handle text files
            try:
                with open(doc, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {doc.name}: {e}")
                continue

        if content:
            langchain_doc = LangchainDocument(page_content=content, metadata={"source": doc.name})

            # Split the document into chunks using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                add_start_index=True,
                separators=["\n\n", "\n", ".", " ", ""],
            )

            docs_processed += text_splitter.split_documents([langchain_doc])

    return docs_processed

def generate_qa_pairs(
    EVAL_CORPUS_PATH: Path,
    N_GENERATIONS: int = 100,
    OUTPUT_PATH: Path = Path("qa_output.xlsx"),
    DOCS_PER_QA: int = 1,  # Number of documents to use for each QA generation
    CHUNKS_PER_DOC: int = 3  # Number of chunks to take from each selected document
) -> List[Dict]:
    """
    Function to generate QA pairs using multiple chunks from one document as context, with a sliding window.

    Parameters:
        EVAL_CORPUS_PATH (Path): The path to the corpus for evaluation.
        N_GENERATIONS (int): The maximum number of QA pairs to generate.
        OUTPUT_PATH (Path): The path where the QA output file will be saved.
        DOCS_PER_QA (int): Number of documents to use per QA generation.
        CHUNKS_PER_DOC (int): Number of chunks to take from each selected document.

    Returns:
        List[Dict]: A list of dictionaries with QA pairs and metadata.
    """
    outputs = []

    # Split documents into chunks
    docs_processed = split_documents(EVAL_CORPUS_PATH)

    # Group chunks by source document
    chunks_by_doc = defaultdict(list)
    for chunk in docs_processed:
        source = chunk.metadata["source"]
        chunks_by_doc[source].append(chunk)

    # Initialize the LLM client
    llm_client = OpenAI_Client()
    logger.info(f"LLM Generation model: {llm_client.model_name}")

    # Ensure we have enough documents to choose from
    doc_names = list(chunks_by_doc.keys())
    if len(doc_names) < DOCS_PER_QA:
        print(f"Warning: Only {len(doc_names)} documents available, but {DOCS_PER_QA} documents requested per QA generation.")

    # Initialize a dictionary to track the current chunk index for each document
    doc_chunk_indices = {doc_name: 0 for doc_name in doc_names}

    # Iterate and generate QA pairs
    for _ in tqdm(range(N_GENERATIONS), total=N_GENERATIONS):
        sampled_contexts = []

        # Choose documents for this QA pair
        selected_docs = doc_names[:DOCS_PER_QA]  # Deterministic choice; can use random.sample() for randomness

        # For each document, sample the specified number of chunks in a sliding window manner
        for doc_name in selected_docs:
            doc_chunks = chunks_by_doc[doc_name]
            start_idx = doc_chunk_indices[doc_name]
            
            # Check if there are enough chunks left for this QA generation
            if start_idx + CHUNKS_PER_DOC > len(doc_chunks):
                print(f"Not enough chunks left in document {doc_name}. Skipping to next document.")
                continue
            
            # Take the specified number of chunks from this document
            sampled_chunks = doc_chunks[start_idx:start_idx + CHUNKS_PER_DOC]
            if not sampled_chunks:  # Check for empty sampled_chunks
                print(f"No chunks available to sample for document {doc_name}. Skipping this document.")
                continue
            
            sampled_contexts.extend(sampled_chunks)

            # Update the index to move the window forward for the next generation
            doc_chunk_indices[doc_name] += CHUNKS_PER_DOC

        # Skip generating QA pair if there are no sampled contexts
        if not sampled_contexts:
            print("No sampled contexts available. Skipping this QA generation.")
            continue

        # Combine chunks into a single context
        combined_context = "\n\n".join(chunk.page_content for chunk in sampled_contexts)

        # Generate QA pair using the combined context
        output_QA_couple = call_llm(llm_client, QA_GENERATION_PROMPT.format(context=combined_context))
        
        try:
            # Extract answers from the LLM output
            simple_question = output_QA_couple.split("Question simple : ")[-1].split("\n")[0].strip()
            simple_answer = output_QA_couple.split("Réponse simple : ")[-1].split("\n")[0].strip()
            reasoning_question = output_QA_couple.split("Question de raisonnement : ")[-1].split("\n")[0].strip()
            reasoning_answer = output_QA_couple.split("Réponse de raisonnement : ")[-1].split("\n")[0].strip()
            multi_context_question = output_QA_couple.split("Question multi-contexte : ")[-1].split("\n")[0].strip()
            multi_context_answer = output_QA_couple.split("Réponse multi-contexte : ")[-1].split("\n")[0].strip()

            # Ensure answers are not too long
            assert len(simple_answer) < 2000, "Answer is too long"
            assert len(reasoning_answer) < 2000, "Answer is too long"
            assert len(multi_context_answer) < 2000, "Answer is too long"

            # Track sources of all chunks
            source_docs = [chunk.metadata["source"] for chunk in sampled_contexts]

            # Append the QA pair to the outputs list
            outputs.append(
                {
                    "context": combined_context,
                    "questions": {
                        "simple": simple_question,
                        "reasoning": reasoning_question,
                        "multi_context": multi_context_question,
                    },
                    "answers": {
                        "simple": simple_answer,
                        "reasoning": reasoning_answer,
                        "multi_context": multi_context_answer,
                    },
                    "source_docs": source_docs,  # Track all source documents
                }
            )
        except Exception as e:
            print(f"Error generating QA pair: {e}")
            continue

    # Save the generated dataset as an Excel file
    qa_df = pd.DataFrame(outputs)
    qa_df.to_excel(OUTPUT_PATH, index=False)
    print(f"Generated QA dataset saved to {OUTPUT_PATH}")

    return outputs  # Return the generated outputs for further processing



# Function to evaluate QA pairs using critique agents
def evaluate_qa_pairs(outputs):
    llm_client = OpenAI_Client()  # Initialize the OpenAI Client for critique
    logger.info(f"LLM Evaluation model: {llm_client.model_name}")

    print("Generating critique for each QA couple...")
    for output in tqdm(outputs):
        try:
            # Initialize evaluation data for this QA pair
            evaluations = {}
            for complexity in ["simple", "reasoning", "multi_context"]:
                question = output["questions"][complexity]
                context = output["context"]

                # Evaluate groundedness
                groundedness_eval = call_llm(
                    llm_client,
                    QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT.format(context=context, question=question),
                )

                # # Evaluate realism
                # realism_eval = call_llm(
                #     llm_client,
                #     QUESTION_REALISM_CRITIQUE_PROMPT.format(context=context, question=question),
                # )

                # # Evaluate standalone quality
                # standalone_eval = call_llm(
                #     llm_client,
                #     QUESTION_STANDALONE_CRITIQUE_PROMPT.format(question=question),
                # )

                # Extract evaluations and scores for groundedness

                evaluation_match = re.search(r"Évaluation :\s*(.*?)\s*Jugement :", groundedness_eval, re.DOTALL)
                score_match = re.search(r"Jugement :\s*([1-5])", groundedness_eval)

                evaluations["groundedness"] = evaluation_match.group(1).strip() if evaluation_match else "Not provided"
                evaluations["groundedness_score"] = int(score_match.group(1)) if score_match else np.nan

                evaluations[f"{complexity}_groundedness"] = groundedness_eval.split("Évaluation : ")[-1].split("Note totale :")[0].strip()
                evaluations[f"{complexity}_groundedness_score"] = groundedness_eval.split("Note totale :")[-1].strip()

                logger.debug(f"faithfulness LLM response: {groundedness_eval}")

                # # Extract evaluations and scores for realism
                # evaluations[f"{complexity}_realism"] = realism_eval.split("Évaluation : ")[-1].split("Note totale :")[0].strip()
                # evaluations[f"{complexity}_realism_score"] = realism_eval.split("Note totale :")[-1].strip()

                # # Extract evaluations and scores for standalone quality
                # evaluations[f"{complexity}_standalone"] = standalone_eval.split("Évaluation : ")[-1].split("Note totale :")[0].strip()
                # evaluations[f"{complexity}_standalone_score"] = standalone_eval.split("Note totale :")[-1].strip()

            # Update the output with evaluations
            output.update(evaluations)

        except Exception as e:
            print(f"Error evaluating QA pair: {e}")
            continue

    return outputs  # Return the updated outputs with evaluations

def extract_numeric(value):
    if value is None or not isinstance(value, str):
        # If the value is None or not a string, return NaN
        return np.nan
    try:
        # Split the value by lines and get the first line
        first_line = value.splitlines()[0].strip()
        # Try to convert the first line to a float if it's numeric
        return float(first_line)
    except (ValueError, IndexError):
        # If conversion fails or no lines, return NaN
        return np.nan


@app.command()
def main(
    eval_corpus_path: Path = Path,  # Default input path
    n_generations: int = 100,  # Number of generations,(It caps when it reaches the maximum chunks possible from the available documents)
    output_path: Path = Path(__file__).parent / "qa_output.xlsx",  # Default output path
    report_output_path: Path = Path(__file__).parent / "report_output.xlsx"  # Default report output path
):
    """
    Function to generate the synthethic data part of the RAG pipeline.
    Currently supports multiple complexity (WIP).
    """
    qa_pairs = generate_qa_pairs(eval_corpus_path, n_generations, output_path)

    if qa_pairs:
        evaluated_pairs = evaluate_qa_pairs(qa_pairs)

        output_data = []
        for output in evaluated_pairs:
            for complexity in ["simple", "reasoning", "multi_context"]:
                output_data.append({
                    "context": output["context"],
                    "question": output["questions"][complexity],
                    "answer": output["answers"][complexity],
                    "complexity": complexity,
                    "source_doc": ", ".join(output["source_docs"]),  # Join multiple sources as a single string
                    "groundedness_evaluation": output.get(f"{complexity}_groundedness"),
                    "groundedness_score": output.get(f"{complexity}_groundedness_score"),
                    # "realism_evaluation": output.get(f"{complexity}_realism"),
                    # "realism_score": output.get(f"{complexity}_realism_score"),
                    # "standalone_evaluation": output.get(f"{complexity}_standalone"),
                    # "standalone_score": output.get(f"{complexity}_standalone_score"),
                })

        # Save the final evaluated QA pairs to an Excel file
        df_output = pd.DataFrame(output_data)
        
        # Apply the function to the score columns
        df_output['groundedness_score'] = df_output['groundedness_score'].apply(extract_numeric)
        # df_output['realism_score'] = df_output['realism_score'].apply(extract_numeric)
        # df_output['standalone_score'] = df_output['standalone_score'].apply(extract_numeric)
        
        df_output.to_excel(report_output_path, index=False)
        print(f"Final evaluated QA dataset saved to {report_output_path}")

        # Filter the rows based on numeric conditions
        filtered_output = df_output[(df_output['groundedness_score'] >= 4)]

        # filtered_output = df_output[
        #     (df_output['groundedness_score'] >= 4) &
        #     (df_output['realism_score'] >= 4) &
        #     (df_output['standalone_score'] >= 4)
        # ]

        # Save the filtered data to an Excel file
        filtered_output = filtered_output[["context", "question", "answer", "complexity", "source_doc"]]
        filtered_output.to_excel(output_path, index=False)
        print(f"Filtered QA dataset saved to {output_path}")

# Run the app
if __name__ == "__main__":
    app()