import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from loguru import logger

# Parameters
MAX_TOKENS = 512

training_args_config_file = ""
series_tokenized_document_path = "./data/processed_data/series_tokenized_paragraphs_asi-gpt-fr-cased-base.pickle"
model_name = "asi/gpt-fr-cased-small"
tokenizer_name = "asi/gpt-fr-cased-base"
output_dir = "./models/"+model_name



# Training config

training_args = TrainingArguments(
    output_dir,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=32,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
)



# Print available GPUs (by default, Hugging Face Trainer uses every available GPUs for training)

available_gpus = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
logger.info(f"Available GPUs: {available_gpus}")



# Load data and split into train and valid

logger.info("Loading tokenized DataFrame from: "+series_tokenized_document_path+"...")
tokenized_documents = pd.read_pickle(series_tokenized_document_path)
logger.info("Loaded")

X_train, X_valid = train_test_split(tokenized_documents, test_size = 0.2)
logger.info(f"Train paragraphs: {len(X_train)}")
logger.info(f"Valid paragraphs: {len(X_valid)}")


# Make PyTorch Dataset

class ParagraphsExtractDataset(Dataset):
    """
    Dataset returning a paragraph from a document, troncated to max_tokens.
    """
    def __init__(self, tokenized_paragraphs, max_tokens):
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.tokenized_paragraphs)

    def __getitem__(self, idx):
        paragraph = self.tokenized_paragraphs.iloc[idx]
        paragraph_len = len(paragraph["input_ids"])

        if paragraph_len < self.max_tokens:
            return paragraph
        else:
            return {
                "input_ids": paragraph["input_ids"][0:self.max_tokens],
                "attention_mask": paragraph["attention_mask"][0:self.max_tokens]
            }



# Load model

model = AutoModelForCausalLM.from_pretrained(model_name)
model_max_input_length = MAX_TOKENS


# Make train and valid Datasets

train_dataset = ParagraphsExtractDataset(X_train, model_max_input_length)
valid_dataset = ParagraphsExtractDataset(X_valid, model_max_input_length)


# Train

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm= False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()

