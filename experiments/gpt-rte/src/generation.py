import pdb

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)




# Parameters

model_name = "./models/asi/gpt-fr-cased-small/checkpoint-9294"
tokenizer_name = "asi/gpt-fr-cased-base"


# Set device on GPU if available, else CPU

available_gpus = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
logging.info(f"Available GPUs: {available_gpus}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# Load model

logging.info(f"Loading model from: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
logging.info("Model loaded")
logging.info(f"Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
logging.info("Tokenizer loaded")


# Generation function

def generate(input_text, max_new_tokens=1):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_text = model.generate(
        input_ids,
        do_sample=True,
        top_p=0.9,
        top_k=20,
        max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(generated_text[0])