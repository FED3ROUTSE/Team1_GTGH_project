class Retriever:
    def __init__(self, embedding_model, vector_store, llm, top_k):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        
    def _retrieve(self, question: str):
        query_embedding = self.embedding_model.embed_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )
        
        return results 
    
    def build_context(self, retrieved_chunks: list[dict]) -> str:
        context_blocks = []

        for item in retrieved_chunks:
            metadata = item["metadata"]

            context_blocks.append(
                f"""
                Source: {metadata["source_file"]}
                Page: {metadata["page_number"]}
                Chunk: {metadata["chunk_index"]}

                {item["content"]}
                """
            )

        return "\n---\n".join(context_blocks)
    
    
    def ask(self, question: str):
        retrieved = self._retrieve(question)
        context = self.build_context(retrieved)

        answer = self.llm.invoke(
            question=question,
            context=context,
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }