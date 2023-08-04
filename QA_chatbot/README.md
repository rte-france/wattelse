# Main idea

QA_chatbot is a directory to explore the implementation of a chatbot that answers questions about a corpus of documents.

There are 3 main steps in the implementation.

## 1. Text extraction

Given a corpus of documents (PDFs, words, websites...), parse them and store them as a list of (short) extracts.

## 2. Information retrieval

Given a list of extracts and a user query, we need to find the top n relevent extracts to answer the query. Each extract and the query are embedded by a model trained to map similar text to similar embeddings (typically SentenceBERT). Cosine similarity is computed between each extract and the query, and top n highest scores are returned.

## 3. Answer generation

Using a generative model (typically instruct models), build a prompt based on relevent extracts to answer user query. A simple prompt could be:

```
Context: concatenation of top n relevent extracts
Question: user query
Instruction: Using the given context, answer the query.
```
