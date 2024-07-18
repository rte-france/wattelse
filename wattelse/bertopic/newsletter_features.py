from pathlib import Path
import os
import locale
from typing import List, Tuple

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

import json

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_newsletter(
    topic_model,
    df: pd.DataFrame,
    topics: List[int],
    df_split: pd.DataFrame,
    top_n_topics: int = None,
    top_n_docs: int = None,
    newsletter_title: str = "Newsletter",
    summarizer_class = None,
    summary_mode: str = "none",
    prompt_language: str = "fr",
    improve_topic_description: bool = False,
    export_base_folder: str = "exported_topics",
    batch_size: int = 10
) -> Tuple[str, str, str, str, str, str]:
    logger.debug("Generating newsletter...")
    
    logger.debug(f"df shape: {df.shape}")
    logger.debug(f"df_split shape: {df_split.shape}")
    logger.debug(f"Number of topics: {len(set(topics))}")

    openai_api = OpenAI_API()
    
    summarizer = summarizer_class() if summary_mode != "none" else None

    # Filter out topic -1 and sort by topic number
    topics_info = topic_model.get_topic_info()
    topics_info = topics_info[topics_info['Topic'] != -1].sort_values('Topic')

    if top_n_topics is None:
        top_n_topics = len(topics_info)
    else:
        top_n_topics = min(top_n_topics, len(topics_info))

    # Date range
    try:
        date_min = df_split[TIMESTAMP_COLUMN].min()
        date_max = df_split[TIMESTAMP_COLUMN].max()
        if pd.isnull(date_min) or pd.isnull(date_max):
            raise ValueError("Invalid timestamp data")
        date_range = f"from {date_min.strftime('%A %d %b %Y')} to {date_max.strftime('%A %d %b %Y')}"
    except Exception as e:
        logger.error(f"Error processing timestamp data: {e}")
        date_range = "Date range unavailable"

    topics_content = []

    for i in tqdm(range(top_n_topics), desc="Processing topics", unit="topic"):
        topic_info = topics_info.iloc[i]
        topic_id = int(topic_info.Topic)
        
        sub_df = get_most_representative_docs(
            topic_model,
            df,
            topics,
            df_split,
            topic_number=topic_id,
            top_n_docs=top_n_docs
        )
        
        if sub_df.empty:
            logger.warning(f"No documents found for topic {topic_id}")
            continue
        
        logger.debug(f"Sub_df for topic {topic_id} shape: {sub_df.shape}")

        topic_title = f"Topic {topic_id}: {', '.join(topic_info['Representation'])}"
        if improve_topic_description:
            improved_topic_description = openai_api.generate(
                FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES.format(
                    title_list=", ".join(sub_df['title'].unique()),
                )
            ).replace('"', "").rstrip('.')
            topic_title = f"Topic {topic_id}: {improved_topic_description}"

        documents = []
        for (title, url, timestamp), group in sub_df.groupby(["title", "url", "timestamp"]):
            try:
                domain = tldextract.extract(url).domain
            except:
                logger.warning(f"Cannot extract URL for document")
                domain = ""
            
            paragraphs = group['text'].tolist()
            
            if summary_mode == 'none':
                content = paragraphs
            elif summary_mode == 'document':
                full_text = "\n\n".join(paragraphs)
                content = [summarizer(full_text, prompt_language=prompt_language)]
            else:  # paragraph mode
                content = [summarizer(para, prompt_language=prompt_language) for para in paragraphs]
            
            doc_info = {
                'title': title,
                'url': url,
                'date': timestamp.strftime('%Y-%m-%d'),
                'domain': domain,
                'paragraphs': content
            }
            documents.append(doc_info)

        topics_content.append({
            'id': topic_id,
            'title': topic_title,
            'documents': documents
        })

    # Generate HTML content
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ newsletter_title }}</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 { 
                color: #2c3e50; 
                border-bottom: 2px solid #2c3e50; 
                padding-bottom: 10px; 
            }
            .topic { 
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                overflow: hidden;
            }
            .topic-header { 
                background-color: #3498db; 
                color: #fff; 
                padding: 10px 15px; 
                cursor: pointer;
                user-select: none;
            }
            .topic-header:hover {
                background-color: #2980b9;
            }
            .topic-content { 
                display: none; 
                padding: 15px;
            }
            .document { 
                border-bottom: 1px solid #eee; 
                padding-bottom: 15px;
                margin-bottom: 15px;
            }
            .document:last-child {
                border-bottom: none;
                margin-bottom: 0;
            }
            .document h3 { 
                margin-top: 0; 
                color: #2c3e50;
            }
            .document .meta { 
                font-size: 0.9em; 
                color: #7f8c8d;
                margin-bottom: 10px;
            }
            a { 
                color: #3498db; 
                text-decoration: none; 
            }
            a:hover { 
                text-decoration: underline; 
            }
            .date-range { 
                font-style: italic; 
                color: #7f8c8d;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>{{ newsletter_title }}</h1>
        <p class="date-range">{{ date_range }}</p>
        {% for topic in topics_content %}
            <div class="topic">
                <div class="topic-header" onclick="toggleTopic({{ topic.id }})">
                    {{ topic.title }}
                </div>
                <div id="topic-{{ topic.id }}" class="topic-content">
                    {% for doc in topic.documents %}
                        <div class="document">
                            <h3><a href="{{ doc.url }}" target="_blank">{{ doc.title }}</a></h3>
                            <p class="meta">{{ doc.date }} | {{ doc.domain }}</p>
                            <p>{{ doc.content }}</p>
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endfor %}

        <script>
            function toggleTopic(topicId) {
                var content = document.getElementById('topic-' + topicId);
                if (content.style.display === 'block') {
                    content.style.display = 'none';
                } else {
                    content.style.display = 'block';
                }
            }
        </script>
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
    html_file_path = os.path.join(export_base_folder, 'newsletter.html')
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Save the JSON file using the custom encoder
    json_file_path = os.path.join(export_base_folder, 'newsletter_data.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(topics_content, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return html_content, date_min, date_max, html_file_path, json_file_path


def export_md_string(newsletter_md: str, path: Path, format="md"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        with open(path, "w") as f:
            f.write(newsletter_md)
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
    df_split,
    topic_number,
    top_n_docs=None
):
    logger.debug(f"Getting most representative docs for topic {topic_number}")
    
    # Filter paragraphs belonging to the specific topic
    topic_mask = np.array(topics) == topic_number
    sub_df = df_split[topic_mask].copy()
    
    logger.debug(f"Number of paragraphs for topic {topic_number}: {len(sub_df)}")

    if sub_df.empty:
        logger.warning(f"No paragraphs found for topic {topic_number}")
        return pd.DataFrame()

    # Group by title, url, and timestamp and count occurrences
    doc_counts = sub_df.groupby(["title", "url", "timestamp"]).size().reset_index(name="paragraph_count")
    doc_counts = doc_counts.sort_values("paragraph_count", ascending=False)

    if top_n_docs is not None:
        doc_counts = doc_counts.head(top_n_docs)

    # Get the paragraphs for these documents
    result = sub_df.merge(doc_counts[["title", "url", "timestamp"]], on=["title", "url", "timestamp"])
    
    logger.debug(f"Number of documents selected for topic {topic_number}: {len(doc_counts)}")
    logger.debug(f"Total paragraphs in selected documents: {len(result)}")

    return result