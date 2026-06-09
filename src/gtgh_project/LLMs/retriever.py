class Retriever:
    def __init__(self, embedding_model, vector_store, llm, top_k, fetch_k, lambda_mult):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        
    def _retrieve(self, question: str):
        query_embedding = self.embedding_model.embed_query(question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            fetch_k = self.fetch_k,
            lambda_mult = self.lambda_mult,
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