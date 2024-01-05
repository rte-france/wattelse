from pathlib import Path
import re

import pandas as pd
import typer
from loguru import logger
from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Dict

from wattelse.common import TEXT_COLUMN, FILENAME_COLUMN

def clean_text(text: str) -> str:
	"""Clean text removing newlines and multiple spaces"""
	text = re.sub("\n", " ", text) # remove newlines
	text = re.sub("\s+", " ", text) # remove multiple spaces
	return text.strip()


def html_table_to_md(html_table: Tag) -> str:
	"""Convert HTML table into MarkDown string"""
	# Get headers
	headers = [clean_text(th.get_text(" ")) for th in html_table.find("tr").find_all(["th", "td"])]

	# Get data for each row
	table_rows = []
	for row in html_table.find_all("tr")[1:]:
		data = [td.get_text(" ") for td in row.find_all("td")]
		if len(data) != len(headers): # TODO: how to handle tables with different row sizes ?
			continue
		else:
			table_rows.append(data)

	# Create Pandas DataFrame and clean it
	df = pd.DataFrame(table_rows, columns=headers)
	df = df.applymap(clean_text)

	return df.to_markdown(index=False)


def html_list_to_md(html_list: Tag) -> str:
	"""Convert HTML list into MarkDown string"""
	md_list = ""
	# Iterate over each bulletpoint
	for line in html_list.find_all("li", recursive=False): # TODO: handle multi-level lists
		md_list += "* " + line.get_text(" ") + "\n"
	return md_list + "\n"


def get_next_tag(tag: Tag) -> Tag:
	"""Return next sibling Tag. If it doesn't exist, recusively return tag parent sibling."""
	next_sibling = tag.find_next_sibling()
	while not next_sibling:
		if tag.parent:
			next_sibling = get_next_tag(tag.parent)
		else:
			break
	return next_sibling


def html_to_md(html_file_path: Path,
			   main_content_tag_name: str = None,
			   main_content_attrs: Dict[str, str] = None
			   ) -> str:
	"""Extract text from html file and transform it to MarkDown string

	Args:
		html_file_path: file path of html to be parsed
		main_content_tag_name: restrict parsing to a specific tag name
		main_content_attrs: restrict parsing to specific tag attributes (id, class...)

	Returns:
		str: _description_
	"""
	# Initialize 
	with open(html_file_path, "r") as file:
		soup = BeautifulSoup(file, "html.parser")
	# Get title
	title = clean_text(soup.find("title").get_text(" "))
	md_content = f"# {title}\n\n"

	# Find main content if provided
	# If not provided, .find(None, None) returns first tag by default
	tag = soup.find(main_content_tag_name, main_content_attrs)

	while tag != None: # iterate tags
		if tag.name =="h1": # if new section
			section_title = clean_text(tag.get_text(" "))
			md_content += (f"# {section_title}\n\n")
			tag = get_next_tag(tag)
		elif tag.name =="h2": # if new sub-section
			section_title = clean_text(tag.get_text(" "))
			md_content += (f"## {section_title}\n\n")
			tag = get_next_tag(tag)
		elif tag.name =="h3": # if new sub-sub-section
			section_title = clean_text(tag.get_text(" "))
			md_content += (f"### {section_title}\n\n")
			tag = get_next_tag(tag)
		elif tag.name =="h4": # if new sub-sub-sub-section
			section_title = clean_text(tag.get_text(" "))
			md_content += (f"#### {section_title}\n\n")
			tag = get_next_tag(tag)
		elif tag.name == "table": # if table
			md_table = html_table_to_md(tag)
			md_content += md_table + "\n\n"
			tag = get_next_tag(tag)
		elif tag.name in ["ul", "ol"]: # if list
			md_list = html_list_to_md(tag)
			md_content += md_list + "\n\n"
			tag = get_next_tag(tag)
		elif tag.name == "p": # if paragraph
			text = clean_text(tag.get_text(" "))
			md_content += text + "\n\n"
			tag = tag = get_next_tag(tag)
		elif tag.name =="div" and not tag.find(): # if div with only text inside
			text = clean_text(tag.get_text(" "))
			md_content += text + "\n\n"
			tag = get_next_tag(tag)
		else: # go to next Tag (sibling or children)
			tag = tag.find_next()
	return md_content
		

def parse_html(html_directory: Path, output_file: Path = "./data/output_md.csv"):
	"""Parse HTML files in a directory"""
	data_dict = {FILENAME_COLUMN: [], TEXT_COLUMN: []}
	for path in Path(html_directory).iterdir():
		if path.is_file() and path.suffix == ".html":
			logger.info(f"Parsing file: {path}")
			data_dict[FILENAME_COLUMN].append(str(path).split("\\")[-1])
			data_dict[TEXT_COLUMN].append(html_to_md(path))
	df = pd.DataFrame(data_dict)
	df.to_csv(output_file)
	logger.info(f"Output stored in: {output_file}")