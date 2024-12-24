# System Architecture

## **Overview**
The PDF Document Semantic Retriever system is designed to extract metadata from PDF documents, generate embeddings, and retrieve semantically similar documents efficiently. The architecture is modular, scalable, and extendable, ensuring adaptability for different use cases such as academic research, document organization, and recommendation systems.


## **High-Level Components**

### 1. **Metadata Extraction**
- **Purpose**: Extracts structured metadata (title, authors, and abstract) from PDFs.
- **Core Functionality**:
  - Utilizes GPT-4o's multimodal vision capabilities to handle diverse document formats.
  - Handles noisy or non-standard PDF layouts to ensure robust metadata extraction.
- **Module**: `src/processing/pdf_reader.py`

### 2. **Embedding Generation**
- **Purpose**: Converts extracted metadata into high-dimensional semantic embeddings for similarity computation.
- **Core Functionality**:
  - Uses the pre-trained `sentence-transformers/all-distilroberta-v1` model for title, authors, and abstract embeddings.
  - Supports additional embedding models for extensibility.
- **Module**: `src/processing/embedding_generator.py`

### 3. **Indexing and Storage**
- **Purpose**: Efficiently stores and organizes embeddings for scalable similarity search.
- **Core Functionality**:
  - FAISS-based indexing for fast nearest-neighbor search.
  - Separate indexes for title, authors, and abstract.
  - Metadata storage synchronized with indexes.
- **Module**: `src/processing/indexing.py`

### 4. **Retrieval Engine**
- **Purpose**: Combines similarity scores across title, authors, and abstract to retrieve the most relevant documents.
- **Core Functionality**:
  - Supports top-k retrieval with configurable relevance weights.
  - Provides API for query handling and result ranking.
- **Module**: `src/retrieval.py`

### 5. **Utilities and Logging**
- **Purpose**: Provides logging and helper functions to ensure maintainability and debuggability.
- **Core Functionality**:
  - Centralized logging for all operations.
  - Utility functions for file I/O and validation.
- **Module**: `src/utils/logger.py`


## **System Workflow**

1. **PDF Input**:
   - A user provides a PDF file to the system.

2. **Metadata Extraction**:
   - The `PDFReader` extracts the title, authors, and abstract using GPT-4's vision capabilities.

3. **Embedding Generation**:
   - The extracted metadata is converted into embeddings via the `EmbeddingGenerator`.

4. **Index Search**:
   - The `Indexing` module performs a similarity search across FAISS indexes for title, authors, and abstract.

5. **Result Ranking**:
   - Similarity scores from title, authors, and abstract are combined with configurable weights.
   - The top-k most relevant documents are returned.

6. **Output**:
   - Metadata and relevance scores of the retrieved documents are presented to the user.

## **Deployment Considerations**

### **Scalability**
- **Indexing**:
  - FAISS supports billions of entries with GPU acceleration.
  - Indexes can be partitioned for distributed storage.

### **Extensibility**
- New embedding models can be integrated by modifying `embedding_generator.py`.
- Custom indexing strategies can be added in `indexing.py`.

### **Fault Tolerance**
- **Metadata Extraction**:
  - Logs detailed errors for failed extractions.
  - Supports retry mechanisms for GPT-4 API calls.
- **Index Management**:
  - Periodic checkpointing to prevent data loss.

### **Performance Optimization**
- Batched processing of PDF files for high throughput.
- GPU acceleration for both embedding generation and FAISS indexing.


## **Technology Stack**

| Component                 | Technology                          |
|---------------------------|--------------------------------------|
| Metadata Extraction       | GPT-4o (OpenAI API)                  |
| Embedding Generation      | `sentence-transformers` (HuggingFace Transformers) |
| Indexing and Retrieval    | FAISS (Facebook AI Similarity Search) |
| Backend Programming       | Python                              |
| Logging                   | Python `logging`                    |


## **Future Enhancements**

1. **Enhanced Metadata Extraction**:
   - Add support for extracting additional fields such as publication date, keywords, and references.

2. **Incremental Index Updates**:
   - Enable real-time updates to the FAISS index without rebuilding.

3. **Web Interface**:
   - Develop a user-friendly interface for uploading PDFs and viewing results.

4. **Distributed Search**:
   - Implement sharded FAISS indexes for large-scale deployments.



