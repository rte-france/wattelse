from typing import List, Optional, Callable

import nltk
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import Transformer, Pooling
from torch import Tensor

from wattelse.summary.lexrank import degree_centrality_scores
from wattelse.summary.summarizer import Summarizer

RTE_MODEL_NAME = "models/language_model_rte/lm-distilcamembertbase-rte-saola2_20220826"
DEFAULT_SUMMARIZER_MODEL = "camembert-base"

DEFAULT_LENGTH_SUMMARY = 3
DEFAULT_RATIO_SUMMARY = 0.1
DEFAULT_CHUNKS_NUMBER_SUMMARY = 6

nltk.download('punkt')

class ExtractiveSummarizer(Summarizer):

    def __init__(self, model_name=DEFAULT_SUMMARIZER_MODEL):
        # Use BERT for mapping tokens to embeddings
        word_embedding_model = Transformer(model_name)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = Pooling(
            (word_embedding_model.get_word_embedding_dimension()),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        # Create sentence transformer model
        self.sentence_transformer_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )


    def generate_summary(self, text, max_length_ratio=DEFAULT_RATIO_SUMMARY) -> str:
        summary = self.summarize_text(text, DEFAULT_LENGTH_SUMMARY, max_length_ratio)
        return " ".join(summary)


    def get_sentences_embeddings(self, sentences: List[str]) -> List[float]:
        """Compute the sentence embeddings"""
        return self.sentence_transformer_model.encode(sentences, convert_to_tensor=True)

    def get_sentences(self, text: str, use_spacy: bool = False) -> List[str]:
        """Return a list of sentences associated to a text (use of sentence tokenizer and some basic filtering)

        Parameters
        ----------
        text : str
            Text from which we want to have sentences extracted
        use_spacy: bool
            Indicates if uses spacy or nltk for sentence tokenization (nltk seems to beave better)

        Returns
        -------
        List[str]
            a list of string containing all relevant sentences

        """
        # filter text according to special tokens __<special>__
        if "__" in text:
            text = text[text.rindex("__") + 2 :]

        if use_spacy:
            nlp_doc = self.nlp(text)

            # extract sentences using spacy nlp object
            sentences = list(filter(lambda t: len(t) > 5, [s.text for s in nlp_doc.sents]))
        else:
            from nltk import sent_tokenize

            sentences = sent_tokenize(text, language="french")
        return sentences

    def get_chunks_embeddings(self, chunks: List[str]) -> List[float]:
        """Constructs chunks embeddings as an average of the embeddings of the sentences contained in the chunks."""
        # TODO: maybe we should start by summarizing each chunk in order to have chunks of same size?
        chunk_embeddings = []
        for chunk in chunks:
            # tokenize each chunk per sentence
            sents = self.get_sentences(chunk)
            emb_sents = self.sentence_transformer_model.encode(sents, convert_to_tensor=True)
            chunk_embeddings.append(torch.mean(emb_sents, 0))
        return torch.stack(chunk_embeddings)

    def summarize_text(
        self,
        text: str,
        max_nb: Optional[int] = DEFAULT_LENGTH_SUMMARY,
        percentage: Optional[float] = DEFAULT_RATIO_SUMMARY,
    ) -> List[str]:
        """Summarizes a text using the maximum number of sentences given as parameter

        Parameters
        ----------
        text : str
            The text to be summarized
        max_nb : Optional[int]
            Maximum number of sentences to include in the summmary.
            Set the value to 0 to ignore the parameter.
        percentage : Optional[float]
            Maximum % of sentences retained from the original text.
            If both max_nb_sentences and percentage are set with inconsistent values, the highest value between both will be chosen.
            Set the value to 0 to ignore the parameter.

        Returns
        -------
        List[str]
            A new text, summary of the input one, represented as a list of sentences
        """
        # Sentence tokenization
        sentences = self.get_sentences(text)

        # Size of summary (max of values represented by max_nb_sentences and percentage in order to avoid conflits)
        summary_size = round(max(max_nb, len(sentences) * percentage))
        logger.debug(f"Maximum size of summary: {summary_size}")
        if len(sentences) <= summary_size:
            logger.warning("Text too small, nothing changed.")
            return sentences

        # Compute the sentence embeddings
        embeddings = self.get_sentences_embeddings(sentences)

        # Computes a summary based on these embeddings
        summary_indices = self.summarize_embeddings(embeddings, summary_size)

        # Export summary as a list of sentences
        summary = [sentences[idx].strip() for idx in summary_indices]
        return summary

    def summarize_chunks(
        self, chunks: List[str], max_nb_chunks: Optional[int] = DEFAULT_CHUNKS_NUMBER_SUMMARY,
    ) -> List[str]:
        """Summarizes a list of text chunks using their embedding representation

        Parameters
        ----------
        chunks : List[str]
            List of text chunks from which we want to select the most representative
        max_nb_chunks : Optional[int]
            Maximum number of chunks to be returned

        Returns
        -------
        List[str]
            a list of selected text chunks

        """
        if len(chunks) <= max_nb_chunks:
            logger.warning("Not enough chunks, nothing changed.")
            return chunks

        # Compute the chunks embeddings
        embeddings = self.get_chunks_embeddings(chunks)

        # Computes a summary based on these embeddings
        summary_indices = self.summarize_embeddings(embeddings, max_nb_chunks)

        # Export summary as a list of chunks
        summary = [chunks[idx] for idx in summary_indices]
        return summary

    def summarize_text_with_additional_embeddings(
        self,
        text: str,
        function_to_compute_embeddings: Callable,
        ratio_for_additional_embeddings: Optional[float] = 0.5,
        max_nb: Optional[int] = DEFAULT_LENGTH_SUMMARY,
        percentage: Optional[float] = DEFAULT_RATIO_SUMMARY,
    ) -> List[str]:
        """Summarizes a text using the maximum number of sentences given as parameter.
        The summary is based on the combination of two embeddings: the "standard" sentence embeddings obtained by
        the sentence_transformers package, and an additional embedding related to the sentences, provided by
        a function given as parameter. The ratio between the two embeddings is governed by an additional parameter.

        Parameters
        ----------
        text : str
            The text to be summarized
        function_to_compute_embeddings: Callable
            A function able to compute embeddings related to a list of sentences
        ratio_for_additional_embeddings: Optional[float]
            Parameter to determine the ratio to be taken into account between the two embeddings: 0 means that the
            additional embedding is not taken into account, 1 means that the default embedding is not taken into account
        max_nb : Optional[int]
            Maximum number of sentences to include in the summmary.
            Set the value to 0 to ignore the parameter.
        percentage : Optional[float]
            Maximum % of sentences retained from the original text.
            If both max_nb_sentences and percentage are set with inconsistent values, the highest value between both will be chosen.
            Set the value to 0 to ignore the parameter.

        Returns
        -------
        List[str]
            A new text, summary of the input one, represented as a list of sentences
        """
        assert 0 <= ratio_for_additional_embeddings
        assert ratio_for_additional_embeddings <= 1

        # Sentence tokenization
        sentences = self.get_sentences(text)

        # Size of summary (max of values represented by max_nb_sentences and percentage in order to avoid conflits)
        summary_size = round(max(max_nb, len(sentences) * percentage))
        logger.debug(f"Maximum size of summary: {summary_size}")
        if len(sentences) <= summary_size:
            logger.warning("Text too small, nothing changed.")
            return sentences

        # Compute the cosine matrix for sentence embeddings
        embeddings = self.get_sentences_embeddings(sentences)
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()

        # Compute the cosine matrix for the additional embeddings
        additional_embeddings = function_to_compute_embeddings(sentences)
        cos_scores_additional = util.pytorch_cos_sim(
            additional_embeddings, additional_embeddings
        ).numpy()

        # Combines the two cosine matrices
        new_cos_scores = (
            ratio_for_additional_embeddings * cos_scores_additional
            + (1 - ratio_for_additional_embeddings) * cos_scores
        )

        # Computes a summary based on the sentence embeddings
        summary_indices = self._summarize_based_on_cos_scores(new_cos_scores, summary_size)

        # Export summary as a list of sentences
        summary = [sentences[idx].strip() for idx in summary_indices]
        return summary

    def _summarize_based_on_cos_scores(self, cos_scores, summary_size: int) -> List[int]:
        """Summarizes "something" on the basis of a cosine similarity matrix.
        This approach may apply to text or a set of chunks.

        Parameters
        ----------
        embeddings : Tensor
            embeddings of 'what' we want to summarize, represented as a Tensor
        summary_size : int
            number of elements to keep in the summary

        Returns
        -------
        List[int]
            the list of indices of elements we want to keep in the summary

        """
        # Computation of centrality scores using the lexrank algorithm
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

        # Argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = np.argsort(-centrality_scores)

        # Cut of results according to summary size
        summary_indices = most_central_sentence_indices[:summary_size]

        # Resort of the indices to avoid mixing the sentences order
        summary_indices.sort()

        return summary_indices

    def summarize_embeddings(self, embeddings: Tensor, summary_size: int) -> List[int]:
        """Summarizes "something" on the basis of its embeddings representation.
        This approach may apply to text or a set of chunks.

        Parameters
        ----------
        embeddings : Tensor
            embeddings of 'what' we want to summarize, represented as a Tensor
        summary_size : int
            number of elements to keep in the summary

        Returns
        -------
        List[int]
            the list of indices of elements we want to keep in the summary

        """
        # Compute the pair-wise cosine similarities
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
        return self._summarize_based_on_cos_scores(cos_scores, summary_size)

    def check_paraphrase(self, sentences: List[str]):
        """Given a list of sentences, returns a list of triplets with the format [score, id1, id2] indicating
        the degree of paraphrase between pairs of sentences."""
        paraphrases = util.paraphrase_mining(self.sentence_transformer_model, sentences)
        return paraphrases

    @staticmethod
    def to_string(summary: List[str]) -> str:
        """Basic display of summary"""
        text = ""
        for s in summary:
            text += s + " [...] "
        else:
            return text
