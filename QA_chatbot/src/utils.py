from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GenerationConfig
from vigogne.preprocess import generate_instruct_prompt


def make_docs_embedding(docs: List[str], embedding_model: SentenceTransformer):
    return embedding_model.encode(docs, show_progress_bar=True)


def extract_n_most_relevant_extracts(n, query, docs, docs_embeddings, embedding_model, similarity_threshold:float = 0):
    query_embedding = embedding_model.encode(query)
    similarity = cosine_similarity([query_embedding], docs_embeddings)[0]

    # Find indices of documents with similarity above threshold
    above_threshold_indices = np.where(similarity > similarity_threshold)[0]

    # Sort above-threshold indices by similarity and select top n
    max_indices = above_threshold_indices[np.argsort(similarity[above_threshold_indices])][-n:][::-1]

    return docs[max_indices].tolist(), similarity[max_indices].tolist()


def generate_answer(instruct_model, tokenizer, query, relevant_extracts, expected_answer_size="short") -> str:
    context = " ".join(relevant_extracts)

    ### MAIN PROMPT ###
    answer_size = "courte" if expected_answer_size=="short" else "détaillée"
    instruct_prompt = f"En utilisant le contexte suivant : {context}\nRépond à la question suivante : {query}. La réponse doit s'appuyer sur le contexte et être {answer_size}."

    prompt = generate_instruct_prompt(instruct_prompt)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        instruct_model.device
    )
    input_length = input_ids.shape[1]
    generated_outputs = instruct_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.0,
            max_new_tokens=512,
        ),
        return_dict_in_generate=True,
    )
    generated_tokens = generated_outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text