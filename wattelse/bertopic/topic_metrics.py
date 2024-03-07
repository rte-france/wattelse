from typing import List

from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
from loguru import logger
from itertools import combinations


def get_coherence_value(topic_model, topics, docs: List[str], coherence_score_type: str="c_npmi") -> float:
    """
    Assess the coherence of topics generated by BERTopic.

    Parameters:
    - topic_model: Bertopic model to be assessed
    - topics (list): List of topic assignments for each document.
    - docs (list): List of documents (texts).
    - coherence_type: coherence score type, to be choosen among  "c_v", "u_mass", "c_uci", "c_npmi"

    Returns:
    - coherence_score (float): Average coherence score for all topics.
    """

    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
                              "ID": range(len(docs)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Use .get_feature_names_out() if you get an error with .get_feature_names()
    words = vectorizer.get_feature_names_out()

    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for topic in range(len(set(topics))-topic_model._outliers):
        words = list(zip(*topic_model.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]

    # Evaluate Coherence
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence=coherence_score_type,
                                     processes=1)
    return coherence_model.get_coherence()




def get_diversity_value(topic_model: BERTopic, topics: List[List[str]], docs: List[str], diversity_score_type: str = "puw", topk: int = 5) -> float:
    """
    Computes the topic diversity based on the specified diversity score type.

    Parameters
    ----------
    - topic_model: Bertopic model to be assessed
    - topics : List[List[str]]
        A list of lists, where each inner list contains words representing a topic.
    - docs (list): List of documents (texts).
    - diversity_score_type : str
        The type of diversity score to compute. Options are 'puw', 'pjd'.

    Returns
    -------
    float
        The computed diversity score.
    """
    
    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
                              "ID": range(len(docs)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    
    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)

    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for topic in range(len(set(topics))-topic_model._outliers):
        words = list(zip(*topic_model.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]
    
    if diversity_score_type == 'puw':
        return proportion_unique_words(topic_words, topk)
    elif diversity_score_type == 'pjd':
        return pairwise_jaccard_diversity(topic_words, topk)
    else:
        raise ValueError(f"Unknown diversity score type: {diversity_score_type}")


def proportion_unique_words(topics, topk):
    """
    compute the proportion of unique words

    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


def pairwise_jaccard_diversity(topics, topk):
    '''
    compute the average pairwise jaccard distance between the topics 
  
    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity
          will be computed
    
    Returns
    -------
    pjd: average pairwise jaccard distance
    '''
    dist = 0
    count = 0
    for list1, list2 in combinations(topics, 2):
        js = 1 - len(set(list1).intersection(set(list2)))/len(set(list1).union(set(list2)))
        dist = dist + js
        count = count + 1
    return dist/count