import pytest
import numpy as np
from processing.pdf_reader import PDFReader
from processing.embedding_generator import EmbeddingGenerator
from processing.indexing import Indexing
import os

# Test PDFReader
def test_read_pdf():
    reader = PDFReader(api_key='')

    metadata = reader.read_pdf("data/query/sample.pdf")
    assert "title" in metadata, "Title is missing in metadata."
    assert "author" in metadata, "Author is missing in metadata."
    assert "abstract" in metadata, "Abstract is missing in metadata."
    assert metadata["title"] != "", "Title should not be empty."
    assert metadata["author"] != "", "Author should not be empty."
    assert metadata["abstract"] != "", "Abstract should not be empty."

# Test EmbeddingGenerator
def test_generate_embedding():
    generator = EmbeddingGenerator(model_name="sentence-transformers/all-distilroberta-v1")
    text = "This is a test sentence for embedding."
    embedding = generator.generate_embedding(text)
    assert embedding.shape == (768,), "Embedding size should match the model dimension."


def test_generate_metadata_embedding():
    generator = EmbeddingGenerator(model_name="sentence-transformers/all-distilroberta-v1")
    metadata = {
        "title": "Test Title",
        "author": "Test Author",
        "abstract": "This is a test abstract for embedding generation."
    }
    embeddings = generator.generate_metadata_embedding(metadata)
    assert "title" in embeddings, "Title embedding is missing."
    assert "author" in embeddings, "Author embedding is missing."
    assert "abstract" in embeddings, "Abstract embedding is missing."
    assert embeddings["title"].shape == (768,), "Title embedding size should match the model dimension."
    assert embeddings["author"].shape == (768,), "Author embedding size should match the model dimension."
    assert embeddings["abstract"].shape == (768,), "Abstract embedding size should match the model dimension."

# Test Indexing
def test_add_and_search():
    embedding_dim = 768
    index = Indexing(embedding_dim=embedding_dim)

    embeddings = [
        {"title": np.random.rand(embedding_dim), "author": np.random.rand(embedding_dim), "abstract": np.random.rand(embedding_dim)},
        {"title": np.random.rand(embedding_dim), "author": np.random.rand(embedding_dim), "abstract": np.random.rand(embedding_dim)},
        {"title": np.random.rand(embedding_dim), "author": np.random.rand(embedding_dim), "abstract": np.random.rand(embedding_dim)},
    ]
    metadata = [
        {"title": "Doc1", "author": "Author1", "abstract": "Abstract1"},
        {"title": "Doc2", "author": "Author2", "abstract": "Abstract2"},
        {"title": "Doc3", "author": "Author3", "abstract": "Abstract3"},
    ]

    for emb, meta in zip(embeddings, metadata):
        index.add_entry(emb, meta)

    query_embeddings = {
        "title": np.random.rand(embedding_dim),
        "author": np.random.rand(embedding_dim),
        "abstract": np.random.rand(embedding_dim),
    }

    results = index.search(query_embeddings, k=2)

    assert len(results) == 2, "Search should return top 2 results."
    for result, score in results:
        assert "title" in result, "Title is missing in the result."
        assert "author" in result, "Author is missing in the result."
        assert "abstract" in result, "Abstract is missing in the result."
        assert isinstance(score, float), "Score should be a float."

if __name__ == "__main__":
    pytest.main(["-v"])
