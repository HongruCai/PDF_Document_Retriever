import faiss
import numpy as np
import os
import json
from src.config import RELEVANCE_WEIGHTS

class Indexing:
    """
    Manages FAISS indexes for different document aspects (title, author, abstract)
    and synchronizes metadata.
    """

    def __init__(self, embedding_dim: int, metadata_file: str = None):
        self.index_title = faiss.IndexFlatL2(embedding_dim)
        self.index_author = faiss.IndexFlatL2(embedding_dim)
        self.index_abstract = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []
        self.metadata_file = metadata_file

        # Load metadata if file is provided
        if metadata_file and os.path.exists(metadata_file):
            self.load_metadata(metadata_file)

    def add_entry(self, embeddings: dict, metadata: dict):
        """
        Add a new entry (embeddings and metadata) to the indexes.

        Args:
            embeddings (dict): Dictionary with keys 'title', 'author', 'abstract'.
            metadata (dict): Metadata associated with the embeddings.
        """
        self.index_title.add(np.array([embeddings["title"]], dtype=np.float32))
        self.index_author.add(np.array([embeddings["author"]], dtype=np.float32))
        self.index_abstract.add(np.array([embeddings["abstract"]], dtype=np.float32))
        self.metadata.append(metadata)

    def search(self, query_embeddings: dict, k: int = 5):
        """
        Search for the most similar entries for title, author, and abstract.

        Args:
            query_embeddings (dict): Query embeddings for 'title', 'author', and 'abstract'.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            list: Combined and ranked metadata from all three indexes.
        """
        dist_title, indices_title = self.index_title.search(np.array([query_embeddings["title"]], dtype=np.float32), k)
        dist_author, indices_author = self.index_author.search(np.array([query_embeddings["author"]], dtype=np.float32), k)
        dist_abstract, indices_abstract = self.index_abstract.search(np.array([query_embeddings["abstract"]], dtype=np.float32), k)

        weights = RELEVANCE_WEIGHTS
        combined_scores = {}

        # Combine scores from title, author, and abstract
        for idx, dist in zip(indices_title[0], dist_title[0]):
            if idx < len(self.metadata):
                combined_scores[idx] = combined_scores.get(idx, 0) + weights["title"] * (1 / (1 + dist))
        for idx, dist in zip(indices_author[0], dist_author[0]):
            if idx < len(self.metadata):
                combined_scores[idx] = combined_scores.get(idx, 0) + weights["author"] * (1 / (1 + dist))
        for idx, dist in zip(indices_abstract[0], dist_abstract[0]):
            if idx < len(self.metadata):
                combined_scores[idx] = combined_scores.get(idx, 0) + weights["abstract"] * (1 / (1 + dist))

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.metadata[idx], score) for idx, score in sorted_results[:k]]

    def save_indexes(self, title_path: str, author_path: str, abstract_path: str):
        """
        Save FAISS indexes to disk.

        Args:
            title_path (str): Path to save the title index.
            author_path (str): Path to save the author index.
            abstract_path (str): Path to save the abstract index.
        """
        os.makedirs(os.path.dirname(title_path), exist_ok=True)
        faiss.write_index(self.index_title, title_path)

        os.makedirs(os.path.dirname(author_path), exist_ok=True)
        faiss.write_index(self.index_author, author_path)

        os.makedirs(os.path.dirname(abstract_path), exist_ok=True)
        faiss.write_index(self.index_abstract, abstract_path)
        

    def load_indexes(self, title_path: str, author_path: str, abstract_path: str):
        """
        Load FAISS indexes from disk.

        Args:
            title_path (str): Path to the title index.
            author_path (str): Path to the author index.
            abstract_path (str): Path to the abstract index.
        """
        self.index_title = faiss.read_index(title_path)
        self.index_author = faiss.read_index(author_path)
        self.index_abstract = faiss.read_index(abstract_path)
        

    def save_metadata(self, metadata_file: str):
        """
        Save metadata to a file.

        Args:
            metadata_file (str): Path to save metadata.
        """
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)
        

    def load_metadata(self, metadata_file: str):
        """
        Load metadata from a file.

        Args:
            metadata_file (str): Path to load metadata from.
        """
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)


