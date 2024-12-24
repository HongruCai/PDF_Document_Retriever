from .processing.pdf_reader import PDFReader
from .processing.embedding_generator import EmbeddingGenerator
from .processing.indexing import Indexing
from .config import (
    METADATA_FILE, INDEX_TITLE_FILE, INDEX_AUTHOR_FILE, INDEX_ABSTRACT_FILE,
    EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K_RESULTS, OPENAI_API_KEY
)
from tqdm import tqdm
from .utils.logger import setup_logger

logger = setup_logger("PDFRetriever", "application.log")

class PDFRetriever:
    """
    A unified class to manage PDF metadata extraction, embedding generation, indexing,
    and document retrieval.
    """

    def __init__(self):
        self.pdf_reader = PDFReader(api_key=OPENAI_API_KEY)
        self.embedding_generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
        self.indexing = Indexing(embedding_dim=EMBEDDING_DIM, metadata_file=None)
        logger.info("PDFRetriever initialized.")

    def initialize_index(self, metadata_file: str):
        """
        Initialize FAISS indexes from a metadata file.

        Args:
            metadata_file (str): Path to the metadata file containing articles.
        """
        logger.info(f"Initializing index from metadata file: {metadata_file}")
        try:
            import json
            with open(metadata_file, "r") as f:
                articles = json.load(f)

            for article in tqdm(articles, desc="Initializing Index"):
                metadata = {
                    "title": article["title"],
                    "authors": article["authors"],
                    "abstract": article["abstract"]
                }
                embeddings = self.embedding_generator.generate_metadata_embedding(metadata)
                self.indexing.add_entry(embeddings, metadata)

            # self.indexing.save_indexes(INDEX_TITLE_FILE, INDEX_AUTHOR_FILE, INDEX_ABSTRACT_FILE)
            # self.indexing.save_metadata(METADATA_FILE)
            logger.info("Index initialized successfully.")
            logger.info(f"Total documents in the index: {len(articles)}")
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise

    def load_index(self):
        """
        Load previously saved FAISS indexes and metadata from disk.
        """
        logger.info("Loading indexes and metadata from disk.")
        try:
            self.indexing.load_indexes(INDEX_TITLE_FILE, INDEX_AUTHOR_FILE, INDEX_ABSTRACT_FILE)
            self.indexing.load_metadata(METADATA_FILE)
            logger.info("Indexes and metadata loaded successfully.")
            logger.info(f"Total documents in the index: {len(self.indexing.metadata)}")
        except Exception as e:
            logger.error(f"Failed to load indexes or metadata: {e}")
            raise

    def add_to_index(self, title: str, authors: str, abstract: str):
        """
        Add a single document's metadata to the index.

        Args:
            title (str): The title of the document.
            authors (str): The author(s) of the document.
            abstract (str): The abstract of the document.
        """
        logger.info(f"Adding document to index: title='{title}'")
        try:
            metadata = {"title": title, "authors": authors, "abstract": abstract}
            embeddings = self.embedding_generator.generate_metadata_embedding(metadata)
            self.indexing.add_entry(embeddings, metadata)
            logger.info("Document added to index successfully.")
            logger.info(f"Total documents in the index: {len(self.indexing.metadata)}")
        except Exception as e:
            logger.error(f"Failed to add document to index: {e}")
            raise

    def save_index(self):
        """
        Save the current state of indexes and metadata to disk.
        """
        logger.info("Saving indexes and metadata to disk.")
        try:
            self.indexing.save_indexes(INDEX_TITLE_FILE, INDEX_AUTHOR_FILE, INDEX_ABSTRACT_FILE)
            self.indexing.save_metadata(METADATA_FILE)
            logger.info("Indexes and metadata saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save indexes or metadata: {e}")
            raise

    def search_by_pdf(self, pdf_path: str, top_k: int = TOP_K_RESULTS):
        """
        Search for the most relevant articles based on the content of a PDF.

        Args:
            pdf_path (str): Path to the PDF file.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: List of the most relevant articles.
        """
        logger.info(f"Searching for similar articles using PDF: {pdf_path}")
        try:
            metadata = self.pdf_reader.read_pdf(pdf_path)
            embeddings = self.embedding_generator.generate_metadata_embedding(metadata)
            results = self.indexing.search(embeddings, k=top_k)
            logger.info(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Failed to search using PDF: {e}")
            raise

    def get_total_documents(self) -> int:
        """
        Get the total number of documents currently encoded in the index.

        Returns:
            int: The total number of documents.
        """
        return len(self.indexing.metadata)




