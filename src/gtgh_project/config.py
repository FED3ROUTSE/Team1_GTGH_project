
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = ROOT / "eu_docs"
VECTOR_DIR = ROOT / "Vector DB Folder"

COLLECTION_NAME = "pdf_rag_collection"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


# For connecting to databricks
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")