import typer
import json
import re
import pandas as pd
from pathlib import Path
from loguru import logger
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from openai import Timeout

from wattelse.api.openai.client_openai_api import OpenAI_Client

BASE_DIR = Path("/DSIA/nlp/ocr-data")
INPUT_DIR = BASE_DIR / "sample_demo"
OUTPUT_DIR = BASE_DIR / "output_upgraded"

app = typer.Typer()


def create_energy_contract_extraction_prompt():
    return """
You are an expert in extracting detailed information from energy contracts and grid connection documents.
Analyze the following document and extract the specific information requested:

Document text:
{content}

Extract and provide the following information in JSON format:
{{
    "substations": [
        {{
            "substation_name": "Name of the substation (Poste)",
            "liaisons": [
                {{
                    "liaison_number": "Liaison number (e.g., 'Liaison 1')",
                    "liaison_type": "Type of liaison (principale, complémentaire, secours)",
                    "liaison_description": "Full description of the liaison",
                    "connection_point": {{
                        "connection_point_number": "Number of the connection point (e.g., 'Point de Connexion n° 1')",
                        "connection_point_details": "Details about the connection point"
                    }},
                    "connection_power": "Power information related to this connection (Puissance de Raccordement)"
                }}
            ]
        }}
    ],
    "evacuation_network": "Evacuation network for producers (réseau d'évacuation)",
    "pst_information": "Points de Surveillance Technique (PST) information if present",
    "contract_details": {{
        "cart_number": "CART contract number (n° de CART)",
        "effective_date": "Effective date of the contract (date de prise d'effet)",
        "accounting_codes": "Accounting codes (codes décomptes)"
    }},
    "client_information": {{
        "client_name": "The full legal name of the company/client",
        "siret": "The SIRET number",
        "vat_id": "The VAT ID number (identifiant TVA)"
    }}
}}

CRITICAL INSTRUCTIONS:
1. IMPORTANT: Each substation (Poste) can have MULTIPLE liaisons, and each liaison can have its own connection point and power information. Preserve this hierarchical relationship.
2. Look for patterns like:
   - "Poste de BATAVIA"
   - "Liaison 1 = constituée par le départ..."
   - "Point de Connexion n° 1 = ..."
   - "Puissance de Raccordement = ..."
3. PST stands for "Points de Surveillance Technique" - look for this term specifically
4. Return ONLY a valid JSON object without ANY additional text
5. Do not include any text before or after the JSON structure
6. The response MUST be parseable by a standard JSON parser
7. If any information is not found in the document, use "Not found" as the value for that field
8. Use only double quotes for JSON properties and values
"""


def clean_and_parse_json(response_text):
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > 0:
                json_text = response_text[start_idx:end_idx]
                return json.loads(json_text)
        except:
            pass

        try:
            fixed_text = re.sub(r"(?<!\w)'([^']*?)'", r'"\1"', response_text)
            return json.loads(fixed_text)
        except:
            pass

        try:
            fixed_text = response_text.replace("'", '"')
            return json.loads(fixed_text)
        except:
            pass

        try:
            json_pattern = r"({[\s\S]*})"
            match = re.search(json_pattern, response_text)
            if match:
                return json.loads(match.group(1))
        except:
            pass

        logger.debug(f"Failed to parse JSON. Raw response: {response_text}")

        return {
            "error": "Failed to parse response as JSON",
            "raw_response": response_text,
        }


def extract_substations_summary(extracted_info):
    if "error" in extracted_info:
        return "Parsing error"

    try:
        substations = extracted_info.get("substations", [])
        if not substations:
            return "No substations found"

        summary_parts = []
        for substation in substations:
            sub_name = substation.get("substation_name", "Unknown")
            liaisons = substation.get("liaisons", [])

            liaison_counts = len(liaisons)
            liaison_info = (
                f"{sub_name} ({liaison_counts} liaisons)"
                if liaison_counts
                else f"{sub_name} (no liaisons)"
            )
            summary_parts.append(liaison_info)

        return ", ".join(summary_parts)
    except Exception as e:
        logger.error(f"Error creating substations summary: {e}")
        return "Error in summary"


def process_file(file_path: Path, prompt_template: str, output_dir: Path):
    llm_client = OpenAI_Client()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        formatted_prompt = prompt_template.format(content=file_content)

        custom_timeout = 120.0
        kwargs = {"max_tokens": 2048, "timeout": Timeout(custom_timeout, connect=10.0)}

        response = llm_client.generate(formatted_prompt, **kwargs)
        extracted_info = clean_and_parse_json(response)

        parsing_error = (
            "error" in extracted_info and "Failed to parse" in extracted_info["error"]
        )

        metadata = {
            "file_name": file_path.name,
            "extraction_status": "success" if not parsing_error else "parsing_error",
            "file_path": str(file_path),
            "extraction_timestamp": pd.Timestamp.now().isoformat(),
        }

        complete_info = {"metadata": metadata, "extracted_data": extracted_info}
        output_file = output_dir / f"{file_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(complete_info, f, indent=4, ensure_ascii=False)

        logger.info(f"Successfully processed file: {file_path.name} -> {output_file}")
        if parsing_error:
            logger.warning(
                f"JSON parsing failed for {file_path.name}, saved raw response"
            )

        summary_info = {
            "file_name": file_path.name,
            "output_file": str(output_file),
            "client_name": "Not found",
            "cart_number": "Not found",
            "substations_summary": "Not found",
            "siret": "Not found",
            "pst_info": "Not found",
            "status": "success" if not parsing_error else "parsing_error",
        }

        try:
            if not parsing_error:
                if "client_information" in extracted_info:
                    client_info = extracted_info["client_information"]
                    summary_info["client_name"] = client_info.get(
                        "client_name", "Not found"
                    )
                    summary_info["siret"] = client_info.get("siret", "Not found")

                if "contract_details" in extracted_info:
                    contract_info = extracted_info["contract_details"]
                    summary_info["cart_number"] = contract_info.get(
                        "cart_number", "Not found"
                    )

                summary_info["pst_info"] = extracted_info.get(
                    "pst_information", "Not found"
                )

                summary_info["substations_summary"] = extract_substations_summary(
                    extracted_info
                )

        except Exception as e:
            logger.error(f"Error extracting summary information: {e}")

        return summary_info

    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {e}")
        return {
            "file_name": file_path.name,
            "output_file": "",
            "client_name": "Not found",
            "cart_number": "Not found",
            "substations_summary": "Not found",
            "siret": "Not found",
            "pst_info": "Not found",
            "status": f"Error: {str(e)}",
        }


def process_all_files(input_dir: Path, output_dir: Path, prompt_template: str):
    text_files = list(input_dir.glob("*.txt"))
    logger.info(f"Found {len(text_files)} text files to process")

    if not text_files:
        logger.warning(f"No text files found in {input_dir}")
        return pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(desc="Processing Files", total=len(text_files)) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(process_file)(file_path, prompt_template, output_dir)
            for file_path in text_files
        )

    return pd.DataFrame(results)


def create_consolidated_json(output_dir: Path):
    consolidated_data = []

    json_files = [
        f
        for f in output_dir.glob("*.json")
        if not f.name.startswith("all_") and not f.name.startswith("summary_")
    ]

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                consolidated_data.append(data)
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    consolidated_path = output_dir / "all_extracted_contracts.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

    return consolidated_path


@app.command()
def main(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    summary_file: str = "processing_summary.json",
    retry_failed: bool = False,
):
    logger.info(f"Using input directory: {input_dir}")
    logger.info(f"Output will be saved to: {output_dir}")

    prompt_template = create_energy_contract_extraction_prompt()
    logger.info(
        "Using specialized energy contract information extraction prompt with hierarchical structure"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    previous_failures = []
    summary_path = output_dir / summary_file
    if retry_failed and summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                previous_summary = json.load(f)
                previous_failures = [
                    item["file_name"]
                    for item in previous_summary
                    if "parsing_error" in item["status"] or "Error" in item["status"]
                ]
            if previous_failures:
                logger.info(
                    f"Found {len(previous_failures)} previously failed files to retry"
                )
        except Exception as e:
            logger.error(f"Error reading previous summary: {e}")

    summary_df = process_all_files(input_dir, output_dir, prompt_template)

    if summary_df.empty:
        logger.warning("No results to save")
        return

    summary_records = summary_df.to_dict(orient="records")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=4, ensure_ascii=False)
    logger.info(f"Processing summary saved to {summary_path}")

    consolidated_path = create_consolidated_json(output_dir)
    logger.info(f"Consolidated data saved to {consolidated_path}")

    success_count = summary_df[summary_df["status"] == "success"].shape[0]
    error_count = summary_df[summary_df["status"] != "success"].shape[0]
    logger.info(
        f"Processing complete: {success_count} successful, {error_count} with errors"
    )


if __name__ == "__main__":
    typer.run(main)
