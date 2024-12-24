from src.retrieval import PDFRetriever

if __name__ == "__main__":
    # Initialize the PDFRetriever
    retriever = PDFRetriever()

    # 1. Initialize Index from Metadata File
    metadata_file = "data/metadata/sampled_1000_papers.json"  
    retriever.initialize_index(metadata_file)

    # 2. Save the Index
    retriever.save_index()

    # 3. Load the Index
    retriever.load_index()

    # 4. Add a New Document to the Index
    new_title = "Large Language Models Empowered Personalized Web Agents"
    new_author = "Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li, Tat-Seng Chua."
    new_abstract = (
    "Web agents have emerged as a promising direction to automate Web task completion based on user instructions, "
    "significantly enhancing user experience. Recently, Web agents have evolved from traditional agents to Large Language Models (LLMs)-based Web agents. "
    "Despite their success, existing LLM-based Web agents overlook the importance of personalized data (e.g., user profiles and historical Web behaviors) "
    "in assisting the understanding of users' personalized instructions and executing customized actions. To overcome the limitation, "
    "we first formulate the task of LLM-empowered personalized Web agents, which integrate personalized data and user instructions "
    "to personalize instruction comprehension and action execution. To address the absence of a comprehensive evaluation benchmark, "
    "we construct a Personalized Web Agent Benchmark (PersonalWAB), featuring user instructions, personalized user data, Web functions, "
    "and two evaluation paradigms across three personalized Web tasks. Moreover, we propose a Personalized User Memory-enhanced Alignment (PUMA) "
    "framework to adapt LLMs to the personalized Web agent task. PUMA utilizes a memory bank with a task-specific retrieval strategy to filter "
    "relevant historical Web behaviors. Based on the behaviors, PUMA then aligns LLMs for personalized action execution through fine-tuning and "
    "direct preference optimization. Extensive experiments validate the superiority of PUMA over existing Web agents on PersonalWAB.")
    retriever.add_to_index(new_title, new_author, new_abstract)

    # 5. Search for Relevant Documents Using a PDF
    query_pdf_path = "data/query/sample.pdf"  
    results = retriever.search_by_pdf(query_pdf_path, top_k=5)
    print("Retrieval Results:-----------------------------------------")
    for idx, (metadata, score) in enumerate(results, start=1):
        print(f"Result {idx}:")
        print(f"  Title: {metadata['title']}")
        print(f"  Author(s): {metadata['authors']}")
        print(f"  Abstract: {metadata['abstract']}")
        print(f"  Relevance Score: {score:.4f}")
        print()


