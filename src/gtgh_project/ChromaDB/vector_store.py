import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.utils import maximal_marginal_relevance as mmr
import numpy as np

class ChromaVectorStore:
    """
    Local Chroma vector Database for storing and retrieving document embeddings.
    """

    def __init__(self, persist_path: str, collection_name: str):
        # Data saved to disk to survive restarts
        # self.client = chromadb.PersistentClient(
        #     path=persist_path,
        #     settings=Settings(anonymized_telemetry=False)
        # )
        self.client = chromadb.HttpClient(host= "chroma-db", port = 8000, database = "chromadb")
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


    def search(self, query_embedding: list[float], top_k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.6):

        """
        Search for similar (to query embedding) chunks
        """
        # Query the collection with the embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],    
            n_results=fetch_k,
            include=["embeddings", "documents", "metadatas", "distances"]
        )

        retrieved = []

        ids = results["ids"][0]                   
        embeddings  = results["embeddings"][0]                                  
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        

        selected_indices = mmr(
            query_embedding=np.array(query_embedding),
            embedding_list=np.array(embeddings),
            lambda_mult=lambda_mult,
            k=top_k
        )
        
        embedding_list = embeddings.tolist()
        
        
        for i in selected_indices:
            i = int(i)
            retrieved.append(
                {
                    "chunk_id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": float(distances[i]),
                    "embedding": embedding_list[i]
                }
            )

        return retrieved
