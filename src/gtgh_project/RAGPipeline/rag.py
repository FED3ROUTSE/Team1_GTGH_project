
# import necessary libraries
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.LLMs.llm_factory import LlmFactory
from src.gtgh_project.LLMs.system_prompt import get_system_prompt
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent.parent
VectorDB_Path = ROOT / "Vector DB Folder"


class RAG:

    def __init__(self):

        # init components
        self.embedding_model = LocalEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm_factory = LlmFactory(localLlm=False, temperature=0)
        self.llm = self.llm_factory.get_llm()

        # connect to vector DB
        persist_path = str(VectorDB_Path)
        collection_name = "vector_db"
        self.vector_store = ChromaVectorStore(persist_path, collection_name)

        # get system prompt
        self.prompt_template = get_system_prompt()


    def query(self, question: str, top_k: int = 5) -> str:

        # vectorize question
        query_embedding = self.embedding_model.embed_query(question)

        # retrieve top k chunks from vector db
        results = self.vector_store.search(query_embedding, top_k=top_k)
        if not results: # safety net
            return "No relevant documents found in the database."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] Document: {result['metadata']['source_file']}\n"
                f"Page: {result['metadata']['page_number']}\n"
                f"Content: {result['content']}\n"
            )

        context = "\n---\n".join(context_parts)

        prompt = self.prompt_template.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAG()

    # Test with a question
    question = "What are some obligations of companies?"

    # Option 1: Simple query
    answer = rag.query(question)
    print(answer)