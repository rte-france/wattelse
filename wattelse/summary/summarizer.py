from abc import ABC, abstractmethod


DEFAULT_SUMMARIZATION_RATIO = 0.2
DEFAULT_MAX_SENTENCES = 3


class Summarizer(ABC):
    @abstractmethod
    def generate_summary(
        self,
        article_text,
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
    ) -> str:
        pass
