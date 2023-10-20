from wattelse.summary.summarizer import Summarizer
from wattelse.summary.text_summarization import TextSummarizer

BASE_PATH = "/home/jerome/dev/origami/rte-origami-nlp-models/rte_origami_nlp_models/.cache" #FIXME!
DEFAULT_SUMMARIZER_MODEL_PATH = "models/language_model_rte/lm-distilcamembertbase-rte-saola2_20220826"


class ExtractiveSummarizer(Summarizer):

    def __init__(self, model_name=DEFAULT_SUMMARIZER_MODEL_PATH):
        summarizer_local_path = BASE_PATH + "/" + model_name
        self.summarizer = TextSummarizer(summarizer_local_path)

    def generate_summary(self, text, max_length_ratio=0.2) -> str:
        summary = self.summarizer.summarize_text(text, 0, max_length_ratio)
        return " ".join(summary)




