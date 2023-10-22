import configparser
import glob
import os

import pandas as pd
import typer
from datetime import datetime

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from wattelse.bertopic.ouput_features import generate_newsletter, export_md_string
from wattelse.bertopic.utils import load_data, parse_literal
from wattelse.bertopic.train import train_BERTopic, EmbeddingModel
from wattelse.bertopic.utils import TIMESTAMP_COLUMN, TEXT_COLUMN

# Config sections
BERTOPIC_CONFIG_SECTION = "bertopic_config"
LEARNING_STRATEGY_SECTION = "learning_strategy"
NEWSLETTER_SECTION = "newsletter"

# Learning strategies
LEARN_FROM_SCRATCH = (
    "learn_from_scratch"  # uses all available data from feed to create the model
)
LEARN_FROM_LAST = "learn_from_last"  # only the last feed data to create the model
INFERENCE_ONLY = "inference_only"  # do not retrain model; reuse existing bertopic model if available, otherwise, fallback to learn_from_scratch for the first run"""

DEFAULT_SCHEDULE = ""


if __name__ == "__main__":
    app = typer.Typer()

    @app.command("newsletter")
    def newsletter_from_feed(
        newsletter_cfg_path: Path = typer.Argument(
            help="Path to newsletter config file"
        ),
        data_feed_cfg_path: Path = typer.Argument(help="Path to data feed config file"),
    ):
        """
        Creates a newsletter associated to a data feed.
        """

        logger.info(f"Reading newsletter configuration file: {newsletter_cfg_path}")

        # read newsletter & data feed configuration
        config = configparser.ConfigParser(converters={"literal": parse_literal})
        config.read(newsletter_cfg_path)
        data_feed_cfg = configparser.ConfigParser()
        data_feed_cfg.read(data_feed_cfg_path)
        learning_strategy = config[LEARNING_STRATEGY_SECTION]
        newsletter_params = config[NEWSLETTER_SECTION]

        # read data
        logger.info(f"Loading dataset...")
        learning_type = learning_strategy.get("learning_strategy", INFERENCE_ONLY)
        model_path = learning_strategy.get("bertopic_model_path", None)
        if learning_type == INFERENCE_ONLY and (
            not model_path or not os.path.isfile(model_path)
        ):
            learning_type = LEARN_FROM_SCRATCH
        """
        dataset = (
            _load_feed_data(data_feed_cfg, learning_type)
            .reset_index(drop=True)
            .reset_index()
        )
        """
        dataset, latest_file = _load_feed_data(data_feed_cfg, learning_type)
        dataset = dataset.reset_index(drop=True).reset_index()
        logger.info(f"Dataset size: {len(dataset.index)}")

        # learn model and predict
        if learning_type == INFERENCE_ONLY:
            topics, topic_model = _load_topic_model(model_path)
        else:
            topics, topic_model = _train_topic_model(config, dataset, latest_file)
            # train topic model with the dataset
            if model_path:
                logger.info(f"Saving topic model to: {model_path}")
                _save_topic_model(
                    topic_model,
                    config.get("topic_model.embedding", "model_name"),
                    model_path,
                )

        # generate newsletter
        logger.info(f"Generating newsletter...")
        newsletter_md = generate_newsletter(
            topic_model,
            dataset,
            topics,
            df_split=None,
            top_n_topics=newsletter_params.getliteral("top_n_topics"),
            top_n_docs=newsletter_params.getliteral("top_n_docs"),
            newsletter_title=newsletter_params.get("title"),
            summarizer_class=newsletter_params.get("summarizer_class"),
        )

        path = (
            newsletter_params.get("output_directory")
            / f"{datetime.today().strftime('%Y-%m-%d')}{newsletter_params.get('id')}"
            / f".{newsletter_params.get('output_format')}"
        )
        export_md_string(
            newsletter_md, path, format=newsletter_params.get("output_format")
        )

    def _train_topic_model(config: configparser.ConfigParser, dataset: pd.DataFrame, latest_file):
        # Step 1 - Embedding model
        embedding_model = EmbeddingModel(
            config.get("topic_model.embedding", "model_name")
        )
        # Step 2 - Dimensionality reduction algorithm
        umap_model = UMAP(**parse_literal(dict(config["topic_model.umap"])))
        # Step 3 - Clustering algorithm
        hdbscan_model = HDBSCAN(**parse_literal(dict(config["topic_model.hdbscan"])))
        # Step 4 - Count vectorizer
        vectorizer_model = CountVectorizer(
            **parse_literal(dict(config["topic_model.count_vectorizer"]))
        )
        # Step 5 - c-TF-IDF model
        ctfidf_model = ClassTfidfTransformer(
            **parse_literal(dict(config["topic_model.c_tf_idf"]))
        )
        # Step 6 - nb topic params
        topic_params = parse_literal(dict(config["topic_model.topics"]))
        if topic_params.get("nr_topics") == 0:
            topic_params["nr_topics"] = None

        topics, _, topic_model = train_BERTopic(
            **topic_params,
            texts=dataset[TEXT_COLUMN],
            indexes=dataset["index"],
            data_name=latest_file, #FIXME pas top!
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            use_cache=False,
            split_by_paragraphs=False,
        )

        return topics, topic_model

    def _load_feed_data(
        data_feed_cfg: configparser.ConfigParser, learning_strategy: str
    ) -> pd.DataFrame:
        data_dir = data_feed_cfg.get("data-feed", "data_dir_path")
        logger.info(f"Loading data from feed dir: {data_dir}")
        list_all_files = glob.glob(
            f"{data_dir}/*.jsonl"
        )
        latest_file = max(list_all_files, key=os.path.getctime)

        if learning_strategy == INFERENCE_ONLY or learning_strategy == LEARN_FROM_LAST:
            # use the last data available in the feed dir
            return load_data(latest_file)

        elif learning_strategy == LEARN_FROM_SCRATCH:
            # use all data available in the feed dir
            dfs = [load_data(f) for f in list_all_files]
            new_df = pd.concat(dfs)
            return new_df, latest_file #FIXME! file name shall be avoided in train

    def _load_topic_model(model_path_dir: str):
        loaded_model = BERTopic.load(model_path_dir)
        return loaded_model

    def _save_topic_model(topic_model, embedding_model, model_path_dir):
        # Method 1 - safetensors
        topic_model.save(
            model_path_dir,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=embedding_model,
        )

    @app.command("install-newsletter")
    def automate_newsletter(
        newsletter_cfg_path: Path = typer.Argument(
            help="Path to newsletter config file"
        ),
        data_feed_cfg_path: Path = typer.Argument(help="Path to data feed config file"),
    ):
        """
        Install in crontab an automatic newsletter creation
        """
        logger.error("Not implemented yet!")

    @app.command("list-newsletters")
    def list_newsletters():
        """
        List from crontab existing automatic newsletters
        """
        logger.error("Not implemented yet!")

    @app.command("remove-newsletter")
    def remove_newsletter(
        newsletter_cfg: Path = typer.Argument(help="Path to newsletter config file"),
    ):
        """
        Removes from crontab an automatic newsletter creation
        """
        logger.error("Not implemented yet!")

    app()
