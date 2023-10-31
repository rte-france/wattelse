from pathlib import Path

# from md2pdf.core import md2pdf
import markdown
import pandas as pd
import tldextract

import wattelse.summary.abstractive_summarizer


def generate_newsletter(
    topic_model,
    df,
    topics,
    df_split=None,
    top_n_topics=5,
    top_n_docs=3,
    top_n_docs_mode="cluster_probability",
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
            domain = tldextract.extract(doc.url).domain
            md_lines.append(f"<div class='timestamp'>{doc.timestamp.strftime('%d-%m-%Y')} | {domain}</div>")
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


def get_most_representative_docs(topic_model, df, topics, mode="cluster_probability", df_split=None, topic_number=0, top_n_docs=3):
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
            .iloc[0:top_n_docs]
        )
        return df[df["title"].isin(sub_df["title"])]

    # If no df_split is None, use mode to determine how to return most representative docs :
    elif mode == "cluster_probability":
        docs_prob = topic_model.get_document_info(df["text"])["Probability"]
        df = df.assign(Probability=docs_prob)
        sub_df = df.loc[pd.Series(topics) == topic_number]
        sub_df = sub_df.sort_values("Probability", ascending=False).iloc[0:top_n_docs]
        return sub_df

    elif mode == "ctfidf_representation":
        # TODO : "get_representative_docs" currently returns maximum 3 docs as implemtented in BERTopic
        # We should modify the function to return more if needed
        docs = topic_model.get_representative_docs(topic=topic_number)
        sub_df = df[df["text"].isin(docs)].iloc[0:top_n_docs]
        return sub_df
