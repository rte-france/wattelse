import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from wattelse.summary.summarizer import (
    Summarizer,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_SUMMARIZATION_RATIO,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# DEFAULT_ABSTRACTIVE_MODEL =  "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
# DEFAULT_ABSTRACTIVE_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
DEFAULT_ABSTRACTIVE_MODEL = "facebook/mbart-large-50"


class AbstractiveSummarizer(Summarizer):
    ## class that performs auto summary using T multi langual model
    def __init__(self, model_name=DEFAULT_ABSTRACTIVE_MODEL):
        self.model_name = model_name
        self.WHITESPACE_HANDLER = lambda k: re.sub(
            "\s+", " ", re.sub("\n+", " ", k.strip())
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(device)

    def generate_summary(
        self,
        article_text,
        max_sentences=DEFAULT_MAX_SENTENCES,
        max_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
    ) -> str:
        return self.summarize_batch([article_text], max_sentences, max_length_ratio)[0]

    def summarize_batch(
        self,
        article_texts: List[str],
        max_sentences: int=DEFAULT_MAX_SENTENCES,
        max_length_ratio: float=DEFAULT_SUMMARIZATION_RATIO,
    ) -> List[str]:
        inputs = self.tokenizer(
            [self.WHITESPACE_HANDLER(text) for text in article_texts],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        input_ids = inputs.input_ids.to(device)

        attention_mask = inputs.attention_mask.to(device)

        max_length = 512#max([round(len(article_text) * max_length_ratio) for article_text in article_texts])

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            no_repeat_ngram_size=2,
            num_beams=4,
        )

        summaries = [self.tokenizer.decode(
            output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) for output_id in output_ids]
        return summaries


if __name__ == "__main__":
    #text = """Les vidéos qui disent que les vaccins approuvés sont dangereux et provoquent l'autisme, le cancer ou l'infertilité font partie de celles qui seront retirées, a indiqué la société. La politique comprend la résiliation des comptes des influenceurs anti-vaccins. Les géants de la technologie ont été critiqués pour ne pas en faire plus pour contrer les fausses informations sur la santé sur leurs sites. En juillet, le président américain Joe Biden a déclaré que les plateformes de médias sociaux étaient en grande partie responsables du scepticisme des gens à se faire vacciner en diffusant des informations erronées, et les a appelés à résoudre le problème. YouTube, qui appartient à Google, a déclaré que 130 000 vidéos avaient été supprimées de sa plateforme depuis l'année dernière, lorsqu'il a mis en place une interdiction de contenu diffusant des informations erronées sur les vaccins Covid. Dans un article de blog, la société a déclaré avoir vu de fausses allégations sur les piqûres de Covid "se transformer en désinformation sur les vaccins en général". La nouvelle politique couvre les vaccins approuvés depuis longtemps, tels que ceux contre la rougeole ou l'hépatite B. "Nous élargissons nos politiques de désinformation médicale sur YouTube avec de nouvelles directives sur les vaccins actuellement administrés qui sont approuvés et confirmés comme sûrs et efficaces par les autorités sanitaires locales. et l'OMS", indique le message, faisant référence à l'Organisation mondiale de la santé."""
    text = "Apollo 11 est une mission du programme spatial américain Apollo au cours de laquelle, pour la première fois, des hommes se sont posés sur la Lune, le lundi 21 juillet 1969. L'agence spatiale américaine, la NASA, remplit ainsi l'objectif fixé par le président John F. Kennedy en 1961 de poser un équipage sur la Lune avant la fin de la décennie 1960. Il s'agissait de démontrer la supériorité des États-Unis sur l'Union soviétique qui avait été mise à mal par les succès soviétiques au début de l'ère spatiale dans le contexte de la guerre froide qui oppose alors ces deux pays. Ce défi est lancé alors que la NASA n'a pas encore placé en orbite un seul astronaute. Grâce à une mobilisation de moyens humains et financiers considérables, l'agence spatiale rattrape puis dépasse le programme spatial soviétique."

    #model_name = "csebuetnlp/mT5_multilingual_XLSum"
    model_name = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"

    summarizer = AbstractiveSummarizer(model_name)
    summary = summarizer.generate_summary(text, 3,0.3)

    print(summary)
