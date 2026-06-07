from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()
ROOT = Path(__file__).parent.parent.parent



"""
INGESTION CONFIG VARIABLES
"""
CELEX_LIST = [
        "32022R2554",
        "32022R0868",
        "32023R2854",
        "32022L2555",
        "32024R1689",
        "32019L1024"
    ]
LANGUAGE = "EN"
FILE_TYPE = "PDF"
DOCS_DIR = ROOT / "eu_docs"
VECTOR_DIR = ROOT / "Vector DB Folder"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200



"""
LLM CONFIG VARIABLES
"""
LOCAL = False
TEMPERATURE = 0.5
TOP_K = 5

LOCAL_AI_API_KEY = os.getenv("LOCAL_AI_API_KEY")
LOCAL_AI_ENDPOINT = os.getenv("LOCAL_AI_ENDPOINT")
LOCAL_AI_MODEL = os.getenv("LOCAL_AI_MODEL")

# For connecting to databricks
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

