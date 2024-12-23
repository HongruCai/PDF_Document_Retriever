
from src.retrieval import PDFRetriever

if __name__ == "__main__":
    retriever = PDFRetriever()
    retriever.load_index()
    results = retriever.search_by_pdf("data/query/sample.pdf")
    print(results)
