import os
import torch

# Paths
PDF_FOLDER = "data/pdfs"                  
METADATA_FILE = "data/metadata/sampled_1000_papers.json"       
INDEX_TITLE_FILE = "data/indexes/title.index"
INDEX_AUTHOR_FILE = "data/indexes/author.index"
INDEX_ABSTRACT_FILE = "data/indexes/abstract.index"


# Model
EMBEDDING_MODEL = "sentence-transformers/all-distilroberta-v1"  
EMBEDDING_DIM = 768
EMBEDDING_TOKEN_LENGTH = 512
MODEL_DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"                                    

# API Key
OPENAI_API_KEY = "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"

SYS_PROMPT = f'''You are a document metadata extraction assistant. 
Based on the provided academic paper, extract its title, author names, and abstract. 
Ensure the abstract is directly taken from the original text, without summarizing or rephrasing. 
Return the result in JSON format with the following keys: 'title', 'authors', 'abstract'.
Stay grounded and do not include any other information.
'''

# Other Configurations
TOP_K_RESULTS = 5  
RELEVANCE_WEIGHTS = {"title": 0.4, "authors": 0.3, "abstract": 0.3}  