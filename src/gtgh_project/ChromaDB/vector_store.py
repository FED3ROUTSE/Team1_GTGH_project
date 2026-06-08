import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """
    Local Chroma vector Database for storing and retrieving document embeddings.
    """

    def __init__(self, persist_path: str, collection_name: str):

        # Data saved to disk to survive restarts
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )
        # Get existing collection, or create new one if it doesn't exist / use cos distance for semantic search
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )


    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):

        """
        Add document chunks and their embeddings to the vector database.
        """

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]

        metadatas = [
            {
                "source_file": chunk["title"],      # changed from chunk["source_file"] to match the result of our
                                                    # splitting implementation
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        # Update if existing, insert if new
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    def search(self, query_embedding: list[float], top_k: int = 5):

        """
        Search for similar (to query embedding) chunks
        """

        # Query the collection with the embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],     # method expects a list of queries, even if only one is passed
            n_results=top_k,
        )

        retrieved = []

        ids = results["ids"][0]                     # method returns a list as well, hence we take the first (and only)
                                                    # result
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Reconstruct each result as a dictionary
        for i in range(len(ids)):
            retrieved.append(
                {
                    "chunk_id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )

        return retrieved
