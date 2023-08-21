# QA chatbot main ideas

QA_chatbot is a directory to explore the implementation of a chatbot that answers questions about a corpus of documents.
The main goal is to use LLM as a way to complement Information Retrieval within documents.


## General approach
There are 3 main steps in the implementation.

### 1. Text extraction

Given a corpus of documents (PDFs, words, websites...), parse them and store them as a list of (short) extracts.

### 2. Information retrieval

Given a list of extracts and a user query, we need to find the top n relevant extracts to answer the query. Each extract and the query are embedded by a model trained to map similar text to similar embeddings (typically SentenceBERT). Cosine similarity is computed between each extract and the query, and top n highest scores are returned.

### 3. Answer generation

Using a generative model (typically instruct models), build a prompt based on relevant extracts to answer user query. A simple prompt could be:

```
Context: concatenation of top n relevant extracts
Question: user query
Instruction: Using the given context, answer the query.
```

## How to run to PoC (proof-of concept)?

### CLI
```
(weak_signals) jerome@linux:~/dev/weak-signals/QA_chatbot$ PYTHONPATH=. python src/chatbot.py --help
                                                                                                                                                      
 Usage: chatbot.py [OPTIONS]                                                                                                                          
                                                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-file        PATH  [default: ./data/BP-2019.csv]                                                                                             │
│ --help                   Show this message and exit.                                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The CLI interface requires to have alread-processed data file.
Data files are `.csv` files that have at least a column `text`.
Additionally, data files may have a column `processed_text`; if this column is present, this one will be considered, otherwise the column `text`.

A number of scripts are provided to extract data from various file formats (PDF, DOCX, MD).
They can also be run in CLI mode.

See `extract_text*.py` for details.

### GUI
To run the PoC in GUI mode (Streamlit app):

```
(weak_signals) jerome@linux:~/dev/weak-signals/QA_chatbot$ PYTHONPATH=. python src/app.py 

```