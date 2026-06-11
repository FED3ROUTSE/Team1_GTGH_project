from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from src.gtgh_project.config import AI_SEARCH_ENDPOINT, AI_SEARCH_API_KEY
from langchain_community.vectorstores.utils import maximal_marginal_relevance as mmr
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import numpy as np

class AzureVectorStore:

    def __init__(self, indexName):

        self.index_client = SearchIndexClient(
            endpoint=AI_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AI_SEARCH_API_KEY)
        )

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")]
        )

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=384,
                vector_search_profile_name="vector-profile"
            ),
            # --- metadata fields ---
            SimpleField(
                name="source_file",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
        ]
        index = SearchIndex(
            name=indexName,
            fields=fields,
            vector_search=vector_search
        )
        self.index_client.create_or_update_index(index)
        self.client = SearchClient(
            endpoint=AI_SEARCH_ENDPOINT,
            index_name=indexName,
            credential=AzureKeyCredential(AI_SEARCH_API_KEY)
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]] ):
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]

        metadatas = [
            {
                "source_file": chunk["title"],  # changed from chunk["source_file"] to match the result of our
                # splitting implementation
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        search_docs = [
            {
                "id": ids[i],
                "content": documents[i],
                "embedding": embeddings[i],
                **metadatas[i]  # unpacks source_file, page_number, chunk_index
            }
            for i in range(len(ids))
        ]

        results = self.client.merge_or_upload_documents(documents=search_docs)

        for result in results:
            if not result.succeeded:
                raise Exception(f"Failed to upsert doc {result.key}: {result.error_message}")

    def search(self, query_embedding: list[float], top_k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.6):
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )

        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "content", "source_file", "page_number", "chunk_index"]
        )

        return [
            {
                "chunk_id": d["id"],
                "content": d["content"],
                "metadata": {
                    "source_file": d["source_file"],
                    "page_number": d["page_number"],
                    "chunk_index": d["chunk_index"],
                },
                "distance": 1 - d["@search.score"],
                "embedding": []
            }
            for d in results
        ]