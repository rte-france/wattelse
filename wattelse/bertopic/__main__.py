import configparser
import glob
import os
from pydoc import locate

import pandas as pd
import typer
from datetime import datetime

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from pathlib import Path

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from wattelse.bertopic.ouput_features import generate_newsletter, export_md_string
from wattelse.bertopic.utils import (
    load_data,
    parse_literal,
    OUTPUT_DIR,
    TEXT_COLUMN,
)
from wattelse.bertopic.train import train_BERTopic, EmbeddingModel
from wattelse.common.vars import FEED_BASE_DIR

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
        if model_path:
            model_path = OUTPUT_DIR / model_path
        if learning_type == INFERENCE_ONLY and (
            not model_path or not model_path.exists()
        ):
            learning_type = LEARN_FROM_SCRATCH

        logger.info(f"Learning strategy: {learning_type}")

        dataset = (
            _load_feed_data(data_feed_cfg, learning_type)
            .reset_index(drop=True)
            .reset_index()
        )

        logger.info(f"Dataset size: {len(dataset.index)}")
        if learning_type == INFERENCE_ONLY:
            # predict only
            topic_model = _load_topic_model(model_path)
            logger.info(f"Topic model loaded from {model_path}")
            logger.info("Computation of embeddings for new data...")
            embeddings = EmbeddingModel(
                config.get("topic_model.embedding", "model_name")
            ).embed(dataset[TEXT_COLUMN])
            topics, probs = topic_model.transform(dataset[TEXT_COLUMN], embeddings)

        else:
            # learn and predict
            topics, topic_model = _train_topic_model(config, dataset)
            # train topic model with the dataset
            if model_path:
                logger.info(f"Saving topic model to: {model_path}")
                _save_topic_model(
                    topic_model,
                    config.get("topic_model.embedding", "model_name"),
                    model_path,
                )

        summarizer_class = locate(newsletter_params.get("summarizer_class"))

        # generate newsletter
        logger.info(f"Generating newsletter...")
        newsletter_md = generate_newsletter(
            topic_model,
            dataset,
            topics,
            df_split=dataset,  # FIXME! check behaviour
            top_n_topics=newsletter_params.getliteral("top_n_topics"),
            top_n_docs=newsletter_params.getliteral("top_n_docs"),
            newsletter_title=newsletter_params.get("title"),
            summarizer_class=summarizer_class,
        )

        output_dir = OUTPUT_DIR / newsletter_params.get("output_directory")

        output_path = (
            output_dir
            / f"{datetime.today().strftime('%Y-%m-%d')}_{newsletter_params.get('id')}"
            f"_{data_feed_cfg.get('data-feed','id')}.{newsletter_params.get('output_format')}"
        )
        export_md_string(
            newsletter_md, output_path, format=newsletter_params.get("output_format")
        )
        logger.info(
            f"Newsletter exported in {newsletter_params.get('output_format')} format: {output_path}"
        )

    def _train_topic_model(config: configparser.ConfigParser, dataset: pd.DataFrame):
        # Step 1 - Embedding model
        embedding_model_name = config.get("topic_model.embedding", "model_name")
        # Step 2 - Dimensionality reduction algorithm
        umap_model = UMAP(**parse_literal(dict(config["topic_model.umap"])))
        # Step 3 - Clustering algorithm
        hdbscan_model = HDBSCAN(**parse_literal(dict(config["topic_model.hdbscan"])))
        # Step 4 - Count vectorizer
        vectorizer_model = CountVectorizer(
            stop_words=stopwords.words(
                config.get("topic_model.count_vectorizer", "stop_words")
            ),
            ngram_range=config.getliteral(
                "topic_model.count_vectorizer", "ngram_range"
            ),
        )
        # Step 5 - c-TF-IDF model
        ctfidf_model = ClassTfidfTransformer(
            **parse_literal(dict(config["topic_model.c_tf_idf"]))
        )
        # Step 6 - nb topic params
        topic_params = parse_literal(dict(config["topic_model.topics"]))
        if topic_params.get("nr_topics") == 0:
            topic_params["nr_topics"] = None

        topic_model, topics, _ = train_BERTopic(
            **topic_params,
            full_dataset=dataset,
            embedding_model_name=embedding_model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            use_cache=False,
        )

        return topics, topic_model

    def _load_feed_data(
        data_feed_cfg: configparser.ConfigParser, learning_strategy: str
    ) -> pd.DataFrame:
        data_dir = data_feed_cfg.get("data-feed", "feed_dir_path")
        logger.info(f"Loading data from feed dir: {FEED_BASE_DIR/data_dir}")
        # filter files according to extension and pattern
        list_all_files = glob.glob(
            f"{FEED_BASE_DIR}/{data_dir}/*{data_feed_cfg.get('data-feed', 'id')}*.jsonl*"
        )
        latest_file = max(list_all_files, key=os.path.getctime)

        if learning_strategy == INFERENCE_ONLY or learning_strategy == LEARN_FROM_LAST:
            # use the last data available in the feed dir
            return load_data(latest_file)

        elif learning_strategy == LEARN_FROM_SCRATCH:
            # use all data available in the feed dir
            dfs = [load_data(f) for f in list_all_files]
            new_df = pd.concat(dfs).drop_duplicates(subset=None, keep="first", inplace=False)
            return new_df

    def _load_topic_model(model_path_dir: str):
        loaded_model = BERTopic.load(model_path_dir)
        return loaded_model

    def _save_topic_model(
        topic_model: BERTopic, embedding_model: EmbeddingModel, model_path_dir: Path
    ):
        full_model_path_dir = OUTPUT_DIR / "models" / model_path_dir
        full_model_path_dir.mkdir(parents=True, exist_ok=True)

        # Serialization using safetensors
        topic_model.save(
            full_model_path_dir,
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
