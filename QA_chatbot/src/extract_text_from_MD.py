from pathlib import Path

import pandas as pd
import typer
from loguru import logger


def extract_paragraphs_with_titles(markdown_text):
    lines = markdown_text.split("\n")

    current_section = None
    paragraphs = []
    paragraph = ""

    for line in lines:
        if line.startswith("#"):  # Section title
            current_section = line.strip("#").strip()
        elif line.strip():  # Non-empty line (paragraph content)
            paragraph += line + "\n"
        elif paragraph:  # Empty line (end of paragraph)
            paragraphs.append(
                {"section_title": current_section, "text": paragraph.strip()}
            )
            paragraph = ""

    if paragraph:  # Handling the last paragraph if it doesn't end with an empty line
        paragraphs.append(
            {"section_title": current_section, "text": paragraph.strip()}
        )

    return paragraphs

def extract_text_from_md(markdown_file_path: Path):
    # Read markdown file content
    with open(markdown_file_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()
        # Extract paragraphs with titles
        paragraphs_with_titles = extract_paragraphs_with_titles(markdown_text)
        # Keep reference in source file
        paragraphs_with_titles = [
            dict(item, **{"file": markdown_file_path.name}) for item in paragraphs_with_titles
        ]
        return paragraphs_with_titles

def parse_mds(md_directory: Path, output_file: Path = "./data/output_md.csv"):
    """Parses a set of .md documents stored in the directory"""
    # Parsed paragraphs
    paragraphs = []

    # Filter only md files
    for path in Path(md_directory).iterdir():
        if path.is_file() and path.suffix == ".md":
            logger.info(f"Parsing file: {path}")
            paragraphs += extract_text_from_md(path)

    pd.DataFrame(paragraphs).to_csv(output_file)
    logger.info(f"Output stored in: {output_file}")


if __name__ == "__main__":
    typer.run(parse_mds)