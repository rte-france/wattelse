import pandas as pd
from pathlib import Path

import pathlib
# from md2pdf.core import md2pdf
from md2html import md2html

import wattelse.summary.abstractive_summarizer


def generate_newsletter(
    _topic_model,
    df,
    topics,
    df_split=None,
    top_n_topics=5,
    top_n_docs=3,
    newsletter_title="Newsletter",
    summarizer_class=wattelse.summary.abstractive_summarizer.AbstractiveSummarizer,
) -> str:
    """
    Write a newsletter using trained BERTopic model.
    """
    # Instantiates summarizer
    summarizer = summarizer_class()

    # Ensure top_n_topics is smaller than number of topics
    topics_info = _topic_model.get_topic_info()[1:]
    if len(topics_info) < top_n_topics:
        top_n_topics = len(topics_info)

    # Store each line in a list
    md_lines = [f"# {newsletter_title}"]
    # Get most represented docs per topic
    for i in range(top_n_topics):
        sub_df = df_split.loc[pd.Series(topics) == i]
        sub_df = (
            sub_df.groupby(["title"])
            .size()
            .reset_index(name="counts")
            .sort_values("counts", ascending=False)
            .iloc[0:top_n_docs]
        )
        sub_df = df[df["title"].isin(sub_df["title"])]
        md_lines.append(
            f"## Sujet {i+1} : {', '.join(topics_info['Representation'].iloc[i])}"
        )
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
        result = md2html.render(md2html.parse_args(None), newsletter_md)
        with open(path, "w") as f:
            f.write(result)
