import argparse
import os
from loguru import logger
from pathlib import Path

from wattelse.common.text_parsers.extract_text_from_MD import parse_md
from wattelse.common.text_parsers.extract_text_from_PDF import parse_pdf
from wattelse.common.text_parsers.extract_text_using_origami import parse_docx

from wattelse.common import BASE_DATA_DIR

parser = argparse.ArgumentParser(description="Initialize document path for user RAG app")
parser.add_argument("-i", "--input_dir", required=True, help="User input files path")
parser.add_argument("-u", "--user_name", required=True, help="User name, parsed doc will be saved into $BASE_DATA_DIR/chatbot/user/user_name/docs")
args = parser.parse_args()

input_dir = Path(args.input_dir)
user_name = Path(args.user_name)
output_dir = Path(BASE_DATA_DIR) / "chatbot/user" / user_name / "docs"

# Create output path if not exist
output_dir.mkdir(parents=True, exist_ok=True) 

# Get all files in input directory
files_list = os.listdir(input_dir)

for file in files_list:
	logger.debug(f"Parsing file : {file}...")
	# Get file format
	extension = file.split('.')[-1]
	# Extract data from file
	if extension == "pdf":
		parse_pdf(input_dir / Path(file), output_dir)
	elif extension == "docx":
		parse_docx(input_dir / Path(file), output_dir)
	elif extension == "md":
		parse_md(input_dir / Path(file), output_dir)
	else:
		logger.error("File type not currently supported")