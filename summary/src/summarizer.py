from abc import ABC, abstractmethod


class Summarizer(ABC):

    DEFAULT_SUMMARIZATION_RATIO = 0.3

    @abstractmethod
    def generate_summary(self, article_text, max_length_ratio=DEFAULT_SUMMARIZATION_RATIO) -> str:
        pass