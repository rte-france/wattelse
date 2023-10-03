from pathlib import Path

import pandas as pd
import typer
from loguru import logger


def extract_paragraphs_with_levels(markdown_text):
    lines = markdown_text.split("\n")

    dict_paragraphs = {
        "level1" : [], # "#" in MD
        "level2" : [], # "##" in MD
        "level3" : [], # "###" in MD
        "paragraph" : [],
    }

    
    for i,line in enumerate(lines[:-1]):
        if line.startswith("#") and line.count("#")==1:  # level 1 section
            level1 = line.strip("#").strip()
            level2 = ""
            level3 = ""
            paragraph = ""
        elif line.startswith("##") and line.count("#")==2:  # level 2 section
            level2 = line.strip("##").strip()
            level3 = ""
            paragraph = ""
        elif line.startswith("###") and line.count("#")==3:  # level 3 section
            level3 = line.strip("###").strip()
            paragraph = ""
        elif line.strip() or lines[i+1].startswith("*"):  # Non-empty line (paragraph content) or next line starts a bullet list
            paragraph += line + "\n"
        elif paragraph:  # Empty line (end of paragraph)
            dict_paragraphs["level1"].append(level1)
            dict_paragraphs["level2"].append(level2)
            dict_paragraphs["level3"].append(level3)
            dict_paragraphs["paragraph"].append(paragraph)
            paragraph = ""

    # Handle last line
    paragraph += lines[-1]
    dict_paragraphs["level1"].append(level1)
    dict_paragraphs["level2"].append(level2)
    dict_paragraphs["level3"].append(level3)
    dict_paragraphs["paragraph"].append(paragraph)
    paragraph = ""

    return pd.DataFrame(dict_paragraphs)

def extract_text_from_md(markdown_file_path: Path):
    # Read markdown file content
    with open(markdown_file_path, "r", encoding="utf-8-sig") as file:
        markdown_text = file.read()
        # Extract paragraphs with titles
        paragraphs_with_levels = extract_paragraphs_with_levels(markdown_text)
        return paragraphs_with_levels

def parse_mds(md_directory: Path, output_file: Path = "./data/output_md.csv") -> Path:
    """Parses a set of .md documents stored in the directory"""
    # DataFrame containing all data
    df = pd.DataFrame(
        {
            "level1" : [], # "#" in MD
            "level2" : [], # "##" in MD
            "level3" : [], # "###" in MD
            "paragraph" : [],
        }
        )

    # Filter only md files
    for path in Path(md_directory).iterdir():
        if path.is_file() and path.suffix == ".md":
            logger.info(f"Parsing file: {path}")
            df = pd.concat([df, extract_text_from_md(path)])
    
    df = df.fillna("")
    # Combine columns to enrich the text
    df["processed_text"] = df.level1 + " | " + df.level2 + " | " + df.level3 + " | " + df.paragraph
    df["processed_text"] = df["processed_text"].str.replace("|  |", "|")

    df.to_csv(output_file)

    logger.info(f"Output stored in: {output_file}")

    return output_file


def parse_md(md_file: Path, output_path: Path) -> Path:
    logger.info(f"Parsing {md_file}...")
    # Parsed paragraphs
    paragraphs = extract_text_from_md(md_file)

    df = pd.DataFrame(paragraphs).fillna("")
    # Combine columns to enrich the text
    df["processed_text"] = (
        "Fichier: " + df.file + "\nTitre: " + df.section_title + "\n" + df.text
    )

    output_file = md_file.stem + ".csv"
    full_output_path = output_path / output_file
    df.to_csv(full_output_path)
    logger.info(f"Saved data file: {full_output_path}")

    return full_output_path


if __name__ == "__main__":
    typer.run(parse_mds)