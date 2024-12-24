import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from src.config import EMBEDDING_MODEL, EMBEDDING_TOKEN_LENGTH, MODEL_DEVICE

def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling to model outputs to get sentence embeddings.
    Follows the Transformers library example.

    Args:
        model_output: The output of the transformer model.
        attention_mask: Attention mask from the tokenizer.

    Returns:
        torch.Tensor: Sentence embeddings after pooling.
    """
    token_embeddings = model_output[0]  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingGenerator:
    """
    A class to handle the generation of embeddings for document metadata using Sentence Transformers.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding generator with a specified model.

        Args:
            model_name (str): The name of the model from Hugging Face Transformers.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(MODEL_DEVICE)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using a Hugging Face model.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The generated embedding as a list of floats.
        """
        if not text.strip():
            return []  

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=EMBEDDING_TOKEN_LENGTH)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_output = mean_pooling(outputs, inputs['attention_mask'])
            normalized_embedding = F.normalize(pooled_output, p=2, dim=1)

        return normalized_embedding.squeeze().tolist()

    def generate_metadata_embedding(self, metadata: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Generate embeddings for the title, author, and abstract of a document's metadata.

        Args:
            metadata (Dict[str, str]): A dictionary containing 'title', 'authors', and 'abstract'.

        Returns:
            Dict[str, List[float]]: A dictionary containing embeddings for each metadata field.
        """
        embeddings = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                embeddings[key] = self.generate_embedding(value)
            elif isinstance(value, list):
                value_str = " ".join(value)
                embeddings[key] = self.generate_embedding(value_str)
            else:
                raise ValueError(f"Unsupported metadata type for key '{key}'.")
        return embeddings


