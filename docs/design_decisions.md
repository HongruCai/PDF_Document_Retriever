# Design Decisions

## **1. Metadata-Based Retrieval**

### **Decision: Use Metadata (Title, Authors, Abstract) for Retrieval**

#### **Rationale**
The choice to rely solely on metadata (title, authors, and abstract) for both the query PDF and the retrieval database was driven by the following considerations:

1. **Practicality and Simplicity**:
   - Extracting full-text content from PDFs is complex and often requires additional tools like OCR or GROBID.
   - Such tools involve separate deployments and dependencies, reducing portability and increasing system complexity.

2. **Challenges in Full-Text Extraction**:
   - Academic articles typically include numerous figures, tables, and references, which introduce significant noise during extraction.
   - Even with successful extraction, the textual data may contain irrelevant sections (e.g., citations, related works) that dilute the effectiveness of retrieval.

3. **Challenges in Full-Text Retrieval**:
   - Using full-text data for retrieval often requires splitting documents into paragraphs or sections, which are then treated as independent retrieval units.
   - For long query documents, a paragraph-wise retrieval strategy can generate many partial matches, complicating the aggregation of results.
   - Sections like "Related Work" or extensive citations in academic papers often skew results towards non-relevant matches.

4. **Focus on Key Information**:
   - In academic research, the title, authors, and abstract are widely recognized as the most critical and representative components of a paper.
   - These elements succinctly capture the core contribution and context of the work, making them ideal for retrieval purposes.

5. **Metadata Extraction from Query PDFs**:
   - To ensure ease of use and portability, GPT-4 Vision (GPT-4o) is employed to extract metadata from query PDFs.
   - The process involves converting the PDF's first page into an image and feeding it to GPT-4o to retrieve the title, authors, and abstract.
   - GPT-4o's advanced multimodal capabilities enable accurate extraction across various document layouts and formats.
   - Compared to traditional OCR-based methods, GPT-4o provides:
     - Higher accuracy in extracting structured data.
     - Minimal inclusion of extraneous noise such as figures or citations.
     - Adaptability to non-standard academic paper formats.

#### **Trade-offs**
- **Advantages**:
  - Simplicity: Minimal processing requirements and no need for complex text parsing.
  - Relevance: Reduced noise compared to full-text-based retrieval.
  - Portability: Eliminates dependencies on additional tools for full-text extraction.
- **Disadvantages**:
  - Potential loss of granularity: Important information within the full text is not directly utilized.
  - Metadata reliance: Retrieval quality heavily depends on the accuracy and completeness of extracted metadata.


## **2. Dense Retrieval Using Metadata**

### **Decision: Employ Dense Retrieval for Metadata Matching**

#### **Rationale**
Having chosen metadata as the primary representation of each document, the next step was to determine an appropriate retrieval strategy. Dense retrieval was selected due to its proven performance and practicality:

1. **Semantic Matching**:
   - Dense retrieval maps metadata fields (title, authors, abstract) into a shared high-dimensional vector space.
   - This enables semantic similarity computations, capturing nuanced relationships beyond exact keyword matching.

2. **Advantages over Traditional Methods**:
   - Compared to BM25 or other sparse retrieval techniques, dense retrieval achieves superior accuracy by leveraging contextual embeddings.
   - BM25 struggles with synonymy and polysemy, where dense retrieval excels due to its semantic understanding.

3. **Practicality Compared to Generative Retrieval**:
   - Generative retrieval, while promising, often requires fine-tuning large models, which is resource-intensive.
   - Dense retrieval benefits from pre-trained models, such as `sentence-transformers/all-distilroberta-v1`, which are optimized for semantic similarity tasks and readily available.

4. **Combining Relevance Across Metadata Fields**:
   - Separate embeddings are generated for the title, authors, and abstract.
   - Relevance is computed independently for each field, and the final score aggregates these individual similarities with configurable weights.
   - This allows the system to prioritize different metadata aspects based on use case (e.g., title-focused vs. abstract-focused retrieval).

#### **Trade-offs**
- **Advantages**:
  - Leverages robust pre-trained models without the need for additional training.
  - Flexible scoring mechanism enables fine-grained relevance tuning.
- **Disadvantages**:
  - Higher computational cost compared to sparse retrieval methods.
  - Pre-trained models may not capture domain-specific nuances without additional fine-tuning.

