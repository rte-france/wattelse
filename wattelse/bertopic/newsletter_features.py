from pathlib import Path
import os
import locale
from typing import List, Tuple

# from md2pdf.core import md2pdf
import markdown
import pandas as pd
import tldextract
from loguru import logger

from wattelse.summary.summarizer import Summarizer
from wattelse.summary.abstractive_summarizer import AbstractiveSummarizer
from wattelse.api.openai.client_openai_api import OpenAI_API
from wattelse.api.prompts import (
    FR_USER_GENERATE_TOPIC_LABEL_TITLE,
    FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES,
    FR_USER_SUMMARY_MULTIPLE_DOCS,
    EN_USER_SUMMARY_MULTIPLE_DOCS,
)
from bertopic._bertopic import BERTopic

# Ensures to write with +rw for both user and groups
os.umask(0o002)

from wattelse.bertopic.utils import TEXT_COLUMN, TIMESTAMP_COLUMN
from jinja2 import Template

from tqdm import tqdm

import pandas as pd
from loguru import logger
import tldextract
from wattelse.bertopic.utils import TEXT_COLUMN, TIMESTAMP_COLUMN
from jinja2 import Template
import os
from wattelse.api.openai.client_openai_api import OpenAI_API
from tqdm import tqdm

def generate_newsletter(
    topic_model,
    df: pd.DataFrame,
    topics: List[int],
    df_split: pd.DataFrame = None,
    top_n_topics: int = None,
    top_n_docs: int = None,
    newsletter_title: str = "Newsletter",
    summarizer_class = None,
    summary_mode: str = "document",
    prompt_language: str = "fr",
    improve_topic_description: bool = False,
    export_base_folder: str = "exported_topics",
    batch_size: int = 10  # New parameter for batching
) -> Tuple[str, str, str, str, str]:
    logger.debug("Generating newsletter...")
    
    openai_api = OpenAI_API()
    
    # Instantiates summarizer
    summarizer = summarizer_class()

    # Filter out topic -1 and sort by topic number
    topics_info = topic_model.get_topic_info()
    topics_info = topics_info[topics_info['Topic'] != -1].sort_values('Topic')

    # Ensure top_n_topics is smaller than number of topics
    if top_n_topics is None:
        top_n_topics = len(topics_info)
    else:
        top_n_topics = min(top_n_topics, len(topics_info))

    # Date range
    try:
        date_min = df[TIMESTAMP_COLUMN].min()
        date_max = df[TIMESTAMP_COLUMN].max()
        if pd.isnull(date_min) or pd.isnull(date_max):
            raise ValueError("Invalid timestamp data")
        date_range = f"from {date_min.strftime('%A %d %b %Y')} to {date_max.strftime('%A %d %b %Y')}"
    except Exception as e:
        logger.error(f"Error processing timestamp data: {e}")
        date_range = "Date range unavailable"

    topics_content = []
    md_lines = [f"# {newsletter_title}", f"<div class='date_range'>{date_range}</div>"]

    # Iterate over topics with progress bar
    for i in tqdm(range(top_n_topics), desc="Processing topics", unit="topic"):
        topic_info = topics_info.iloc[i]
        topic_id = topic_info.Topic
        
        sub_df = get_most_representative_docs(
            topic_model,
            df,
            topics,
            mode="cluster_probability",
            df_split=df_split,
            topic_number=topic_id,
            top_n_docs=top_n_docs
        )

        # Batch processing for document summarization
        if summary_mode == 'document':
            summaries = []
            for start_idx in range(0, len(sub_df), batch_size):
                batch_df = sub_df.iloc[start_idx:start_idx+batch_size]
                texts = [doc.text for _, doc in batch_df.iterrows()]
                batch_summaries = summarizer.summarize_batch(texts, prompt_language=prompt_language)
                summaries.extend(batch_summaries)
        elif summary_mode == 'topic':
            article_list = ""
            for _, doc in sub_df.iterrows():
                article_list += f"Titre : {doc.title}\nContenu : {doc.text}\n\n"
            
            topic_summary = openai_api.generate(
                (FR_USER_SUMMARY_MULTIPLE_DOCS if prompt_language=='fr' else EN_USER_SUMMARY_MULTIPLE_DOCS).format(
                    keywords=', '.join(topic_info['Representation']),
                    article_list=article_list,
                )
            )
        else:
            logger.error(f'{summary_mode} is not a valid parameter for argument summary_mode in function generate_newsletter')
            return None

        topic_title = f"Topic {topic_id}: {', '.join(topic_info['Representation'])}"
        if improve_topic_description:
            improved_topic_description = openai_api.generate(
                FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES.format(
                    title_list=(" ; ".join(summaries) if summary_mode=='document' else topic_summary),
                )
            ).replace('"', "").rstrip('.')
            topic_title = f"Topic {topic_id}: {improved_topic_description}"

        md_lines.append(f"## {topic_title}")
        if summary_mode == 'topic':
            md_lines.append(topic_summary)

        documents = []
        for j, (_, doc) in enumerate(sub_df.iterrows()):
            try:
                domain = tldextract.extract(doc.url).domain
            except:
                logger.warning(f"Cannot extract URL for {doc}")
                domain = ""
            
            md_lines.append(f"### [*{doc.title}*]({doc.url})")
            md_lines.append(f"<div class='timestamp'>{doc.timestamp.strftime('%A %d %b %Y')} | {domain}</div>")
            if summary_mode == 'document':
                md_lines.append(summaries[j])
            
            documents.append({
                'title': doc.title,
                'url': doc.url,
                'date': doc.timestamp.strftime('%A %d %b %Y'),
                'domain': domain,
                'summary': summaries[j] if summary_mode == 'document' else ''
            })

        topics_content.append({
            'title': topic_title,
            'summary': topic_summary if summary_mode == 'topic' else '',
            'documents': documents
        })

    # Generate Markdown content
    md_content = "\n\n".join(md_lines)

    # Generate HTML content
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ newsletter_title }}</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #ff0000; border-bottom: 2px solid #ff0000; padding-bottom: 10px; }
            h2 { color: #0000ff; margin-top: 30px; }
            .date-range { font-style: italic; color: #666; }
            .document { background-color: #f9f9f9; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
            .document h3 { margin-top: 0; }
            .document .meta { font-size: 0.9em; color: #666; }
            a { color: #0066cc; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>{{ newsletter_title }}</h1>
        <p class="date-range">{{ date_range }}</p>
        {% for topic in topics_content %}
            <h2>{{ topic.title }}</h2>
            {% if topic.summary %}
                <p>{{ topic.summary }}</p>
            {% endif %}
            {% for doc in topic.documents %}
                <div class="document">
                    <h3><a href="{{ doc.url }}">{{ doc.title }}</a></h3>
                    <p class="meta">{{ doc.date }} | {{ doc.domain }}</p>
                    {% if doc.summary %}
                        <p>{{ doc.summary }}</p>
                    {% endif %}
                </div>
            {% endfor %}
        {% endfor %}
    </body>
    </html>
    """

    template = Template(html_template)
    html_content = template.render(
        newsletter_title=newsletter_title,
        date_range=date_range,
        topics_content=topics_content
    )

    # Save the HTML file
    os.makedirs(export_base_folder, exist_ok=True)
    file_path = os.path.join(export_base_folder, 'newsletter.html')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return md_content, html_content, date_min, date_max, file_path

def export_md_string(newsletter_md: str, path: Path, format="md"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        with open(path, "w") as f:
            f.write(newsletter_md)
    # elif format == "pdf":
    #    md2pdf(path, md_content=newsletter_md)
    elif format == "html":
        result = md2html(newsletter_md, Path(__file__).parent / "newsletter.css")
        with open(path, "w") as f:
            f.write(result)


def md2html(md: str, css_style: Path = None) -> str:
    html_result = markdown.markdown(md)
    if not css_style:
        return html_result
    output = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <style type="text/css">
    """
    cssin = open(css_style)
    output += cssin.read()
    output += """
        </style>
    </head>
    <body>
    """
    output += html_result
    output += """</body>
    </html>
    """
    return output


def get_most_representative_docs(
    topic_model,
    df,
    topics,
    mode="cluster_probability",
    df_split=None,
    topic_number=0,
    top_n_docs=None
):
    """
    Return most representative documents for a given topic.

    - If df_split is not None :
        Groups splited docs by title to count how many paragraphs of the initial document belong to the topic.
        Returns docs having the most occurences.

    - If df_split is None:
        Uses mode to determine the method used. Currently support :
            * cluster_probability : computes the probability for each docs to belong to the topic using the clustering model. Returns most likely docs.
            * ctfidf_representation : computes c-TF-IDF representation for each docs and compare it to topic c-TF-IDF vector using cosine similarity. Returns highest similarity scores docs.

    """
    # If df_split is not None :
    if isinstance(df_split, pd.DataFrame):
        # Filter docs belonging to the specific topic
        sub_df = df_split.loc[pd.Series(topics) == topic_number]
        # Most representative docs in a topic are those with the highest number of extracts in this topic
        sub_df = (
            sub_df.groupby(["title"])
            .size()
            .reset_index(name="counts")
            .sort_values("counts", ascending=False)
        )
        if top_n_docs is not None:
            sub_df = sub_df.head(top_n_docs)
        return df[df["title"].isin(sub_df["title"])]

    # If no df_split is None, use mode to determine how to return most representative docs :
    elif mode == "cluster_probability":
        docs_prob = topic_model.get_document_info(df["text"])["Probability"]
        df = df.assign(Probability=docs_prob)
        sub_df = df.loc[pd.Series(topics) == topic_number]
        sub_df = sub_df.sort_values("Probability", ascending=False)
        if top_n_docs is not None:
            sub_df = sub_df.head(top_n_docs)
        return sub_df

    elif mode == "ctfidf_representation":
        # Get all documents for the topic
        docs = topic_model.get_representative_docs(topic=topic_number)
        sub_df = df[df["text"].isin(docs)]
        if top_n_docs is not None:
            sub_df = sub_df.head(top_n_docs)
        return sub_df