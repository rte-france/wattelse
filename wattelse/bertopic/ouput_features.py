import pandas as pd
from pathlib import Path

# from md2pdf.core import md2pdf
import markdown
import wattelse.summary.abstractive_summarizer
from wattelse.bertopic.app.app_utils import get_most_representative_docs


def generate_newsletter(
    topic_model,
    df,
    topics,
    df_split=None,
    top_n_topics=5,
    top_n_docs=3,
    top_n_docs_mode="ctfidf_representation",
    newsletter_title="Newsletter",
    summarizer_class=wattelse.summary.abstractive_summarizer.AbstractiveSummarizer,
) -> str:
    """
    Write a newsletter using trained BERTopic model.
    """
    # Instantiates summarizer
    summarizer = summarizer_class()

    # Ensure top_n_topics is smaller than number of topics
    topics_info = topic_model.get_topic_info()[1:]
    if len(topics_info) < top_n_topics:
        top_n_topics = len(topics_info)

    # Store each line in a list
    md_lines = [f"# {newsletter_title}"]
    # Iterate over topics
    for i in range(top_n_topics):
        md_lines.append(
            f"## Sujet {i+1} : {', '.join(topics_info['Representation'].iloc[i])}"
        )
        sub_df = get_most_representative_docs(topic_model, df, topics, mode=top_n_docs_mode, df_split=df_split, topic_number=i, top_n_docs=top_n_docs)            
        for _, doc in sub_df.iterrows():
            # Generates summary for article
            summary = summarizer.generate_summary(doc.text)

            # Write newsletter
            md_lines.append(f"### [*{doc.title}*]({doc.url})")
            # FIXME handles special columns if needed
            # md_lines.append(f"{doc.domain} | {doc.timestamp.strftime('%d-%m-%Y')}")
            md_lines.append(summary)

    # Write full file
    md_content = "\n\n".join(md_lines)
    return md_content

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
