import torch
from vigogne.preprocess import generate_instruct_prompt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GenerationConfig


def make_docs_embedding(docs, embedding_model):
    return embedding_model.encode(docs)


def extract_n_most_relevent_extracts(n, query, docs, docs_embeddings, embedding_model):
    query_embedding = embedding_model.encode(query)
    similarity = cosine_similarity([query_embedding], docs_embeddings)[0]
    max_index = similarity.argsort()[-n:][::-1]
    return docs[max_index].tolist()
    

def generate_answer(instruct_model, tokenizer, query, relevent_extracts):
    context = " ".join(relevent_extracts)

    ###┘ MAIN PROMPT ###

    instruct_prompt = f"En utilisant le contexte suivant : {context}\nRépond à la question suivante : {query}. La réponse doit s'appuyer sur le contexte et être détaillée."
    
    ####################

    prompt = generate_instruct_prompt(instruct_prompt)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(instruct_model.device)
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
    print(generated_text)