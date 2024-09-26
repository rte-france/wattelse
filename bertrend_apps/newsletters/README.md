# BERTopic

This folder contains the code for the topic modeling part of BERTrend. It mainly uses [BERTopic](https://github.com/MaartenGr/BERTopic), a topic modeling technique that uses BERT embeddings to create topics.

## Topic modeling using BERTopic

The [app](app) folder contains a Streamlit app that allows to train a BERTopic model and visualize topics. To run the app, use the following command:

```bash
streamlit run app/Main_page.py
```

The app allows you to train a BERTopic model on a given dataset, visualize the topics, generate a newsletter and show a simple representation of topic evolution over time.

## Weak signals detection

The [weak_signals](weak_signals) folder contains tools to further study the temporal evolution of topics and detect weak signals in a large dataset. A Streamlit app allows to try the weak signals detection alogrithm on a given dataset. To run the app, use the following command:

```bash
streamlit run weak_signals/app.py
```

## Newsletter generation

It is possible to generate a newsletter based on the topics found by a BERTopic model using LLM to summarize topics. There are currently two ways of generating newsletter.

### Configuration files

To generate a newsletter, you should first create the following configuration files:

- `newsletter.cfg`: sets parameters for the newsletter and the topic model. You can follow examples in [bertrend_apps/config/newsletters](../config/newsletters).
- `feed.cfg`: define on which feed you want to generate the newsletter and at which frequency. You can follow examples in [bertrend_apps/config/feeds](../config/feeds).

### One-shot creation

Generate just one newsletter based on a specific dataset. You can use the following command to show help:

```bash
python -m bertrend_apps.newsletters newsletter --help
```

Then create a single newsletter based on your configuration files:
```bash
python -m bertrend_apps.newsletters newsletter newsletter.cfg feed.cfg
```

### Scheduled creation

To automatically send a newsletter at regular intervals, you should define a job that scraps your feed and a job that generated the newsletter.

#### New automatic feed installation

Follow this example:

```bash
python -m bertrend_apps.data_provider schedule-scrapping feed.cfg
```

#### New automatic newsletter installation

Follow this example:

```bash
python -m bertrend_apps.newsletters schedule-newsletter newsletter.cfg feed.cfg
```