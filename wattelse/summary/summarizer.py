from abc import ABC, abstractmethod
from typing import List

DEFAULT_SUMMARIZATION_RATIO = 0.2
DEFAULT_MAX_SENTENCES = 3


class Summarizer(ABC):
    @abstractmethod
    def generate_summary(
        self,
        article_text: str,
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
    ) -> str:
        pass

    @abstractmethod
    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
    ) -> List[str]:
        return [self.generate_summary(t) for t in article_texts]
