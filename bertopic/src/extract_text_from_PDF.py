import fitz #pymupdf
import pandas as pd
import re

import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


### Parameters

pdf_file_path = "./data/memento_surete_2004_complet__.pdf"
save_path = "./data/suerete_2004.csv"
filter_text_value = 20 # blocks with less than filter_text_value words will be discarded


### Load pdf and parse blocks of text

data_dict = {"text":[], "page_number":[]}

with fitz.open(pdf_file_path) as doc:
    for page in doc:
        text = page.get_text("blocks")
        for elem in text:
            data_dict["text"].append(elem[4]) # store text only
            data_dict["page_number"].append(page.number)
            

### Process collected blocks

# Transform into a pandas DataFrame

df = pd.DataFrame.from_dict(data_dict)
logging.info(f"Found {len(df)} blocks of text")

# Clean text

def clean_text(x):
    """
    Function applied to clean the text column in the DataFrame. 
    """
    
    # Weird behavior of pymupdf
    x = re.sub("ff ", "ff", x)
    x = re.sub("ﬁ ", "fi", x)
    x = re.sub("fi ", "fi", x)
    x = re.sub("�", " ", x)
    
	# Remove block structure
    x = re.sub("-\n", "", x)
    x = re.sub("\n", " ", x)
    x = re.sub(" +", " ", x)
    x = x.strip()

    return x

df["text"] = df["text"].apply(clean_text).astype(str)

# Filter blocks to keep paragraphs only

logging.info(f"Removing blocks having less than {filter_text_value} words...")
df = df[df["text"].str.split().apply(len)>filter_text_value].reset_index(drop=True)
logging.info(f"{len(df)} blocks remaining")

### Save 

df.to_csv(save_path)
logging.info(f"Saved DF to {save_path}")


        