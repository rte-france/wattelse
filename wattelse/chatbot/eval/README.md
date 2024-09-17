# RAG evaluation

## Overview
This folder contains the main function for RAG evaluation.

The evaluation pipeline currently evaluates the generation part of the RAG only.

The metrics used are:
- **BERTScore**, using the official [https://github.com/Tiiiger/bert_score](bert-score) implementation. As specified [here](https://github.com/Tiiiger/bert_score?tab=readme-ov-file#default-behavior), the default model used for french is `bert-base-multilingual-cased`.
- **LLM as a judge**, using a prompt defined in [wattelse/chatbot/eval/prompt.py](prompt.py). The LLM used for evaluating answers can be specified using environment variables (see [wattelse/api/openai](../api/openai))

## Usage

Go to the eval folder:

```bash
cd wattelse/chatbot/eval
```

Run this command to show infos:

```bash
python main.py --help
```

The script takes two mandatory arguments as input:
- `qr_df_path`: path to the evaluation QR dataframe. The file must be a `.xlsx` file. It must contains the columns:
    - `query`: the query to be evaluated
    - `answser`: the ground truth
    - `doc_list`: the document list the RAG should use to generate the answer, in a python list format, e.g. `["doc1.pdf", "doc2.docx", ...]`

    See [here](https://rtefrance.sharepoint.com/:x:/r/sites/Signauxfaibles/Shared%20Documents/General/RAG%20Evaluation/Corpus%20evaluation/Eval_BE/QR_BE.xlsx?d=w9098383374274af594c80c233d397725&csf=1&web=1&e=DQWyZm) for an example file.
- `eval_corpus_path`: path to the evaluation corpus folder. The folder must contain all documents needed for the evaluation, i.e. all documents listed in the `doc_list` column of the QR dataframe.

Example usage:
```bash
python main.py /path/to/qr_df.xlsx /path/to/eval_corpus/
```

You can specifiy the path to the output result using the `--output_path` argument, for example:

```bash
python main.py /path/to/qr_df.xlsx /path/to/eval_corpus/ --output_path /path/to/output.xlsx
```