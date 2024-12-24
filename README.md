# PDF Document Retriever

## **Overview**

The PDF Document Retriever is a system designed to extract metadata from PDF files (title, authors, and abstract), generate embeddings for these metadata fields, and efficiently retrieve semantically similar documents from a database. It is ideal for applications such as research collaboration, document organization, and expert recommendation.


## **Key Features**


1. **PDF Metadata Extraction**:
   - Leverages GPT-4's multimodal vision capabilities to extract title, authors, and abstract directly from the PDF.
   - Adapts seamlessly to a wide variety of academic paper formats, ensuring high accuracy in metadata extraction.

2. **Embedding Generation**:
   - Converts extracted metadata fields into semantic embeddings using a pre-trained language model (`sentence-transformers/all-distilroberta-v1`).
   - Produces high-quality embeddings for titles, authors, and abstracts, enabling precise document retrieval.

3. **Indexing and Retrieval**:
   - Builds FAISS indexes for efficient nearest-neighbor search across titles, authors, and abstracts.
   - Combines similarity scores from multiple metadata fields to support retrieval of top-k most similar documents.

4. **Extensibility**:
   - Easily extendable for new models, indexing strategies, or document types.
   - Supports additional metadata extraction methods and customized retrieval algorithms.


## **Directory Structure**

```
PDR/
├── data/
│   ├── metadata/                      # Metadata files for documents
│   ├── query/                         # Sample query PDFs
|   ├── index/                         # Index files
|   ├── logs/                          # Log files
├── docs/                              
|   |── architecture.md                # System architecture
|   |── design_decisions.md            # Design decisions
├── src/
│   ├── processing/
│   │   ├── pdf_reader.py              # Extract metadata from PDFs
│   │   ├── embedding_generator.py     # Generate embeddings for metadata
│   │   ├── indexing.py                # Manage FAISS indexing
│   ├── utils/
│   │   ├── logger.py                  # Logging utilities
│   │   ├── test_functions.py          # Utility functions for testing
│   ├── config.py                      # Configuration settings
│   ├── retrieval.py                   # Core retrieval logic (PDFRetriever class)
├── run.py                             # Entry point to demonstrate system functionality
├── requirements.txt                   # Required dependencies
```


## **Installation**


### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/HongruCai/PDF_Document_Retriever.git
   cd /PDF_Document_Retriever
   ```

2. Create a conda environment and activate it:
   ```bash
   conda create -n pdr python=3.11
   conda activate pdr
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that FAISS is installed:
   ```bash
   conda install faiss-cpu
   ```


## **Usage**

### Running the Demonstration
Before running the demonstration, ensure to fill your openai API key in `config.py` to use the gpt-4o model for metadata extraction.

To demonstrate the system's capabilities:
```bash
python run.py
```
This will:
1. Initialize an index from a sample metadata file.
2. Save the index.
2. Load the index.
3. Add a new document to the index.
4. Search for similar documents using a sample PDF.


## **Customization**

- **Configurable Settings**:
  - Modify `config.py` to change default file paths, embedding model, or indexing dimensions.

- **Extending the System**:
  - Add new metadata extraction logic in `pdf_reader.py`.
  - Replace or fine-tune the embedding model in `embedding_generator.py`.
  - Use a different similarity metric or indexing strategy in `indexing.py`.


## **Testing**

Run the test suite to verify functionality:
```bash
pytest src/utils/test_functions.py -v
```


## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

