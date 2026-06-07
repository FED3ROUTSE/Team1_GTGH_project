from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.splitter_factory import SplitterFactory
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (EMBEDDING_MODEL_NAME, VECTOR_DIR,
COLLECTION_NAME, CELEX_LIST, DOCS_DIR, VECTOR_DIR, FILE_TYPE)
import os 
import requests
from pathlib import Path

def run_ingestion():
    failed_downloads = []
    downloader = EurLexDownloader(file_type = FILE_TYPE)

    for celex in CELEX_LIST:
        try:
            downloader.download(celex)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download file with celex_id: {celex}")
            failed_downloads.append(celex)

    splitter = SplitterFactory(downloader.file_type).get_splitter()

    file_chunks = []
    texts = []

    for file in os.listdir(DOCS_DIR):
        name, extension = file.split(".")
        if extension!=downloader.file_type.lower() or name[:-3] not in CELEX_LIST:
            continue
        try:
            file_chunks += splitter.split(file_path = str(DOCS_DIR).split("\\")[-1] + "\\" + file)
        except: 
            print(f"File {file} could not be split")
    
    texts.extend(chunk["content"] for chunk in file_chunks)
    
    model = LocalEmbeddingModel(EMBEDDING_MODEL_NAME)
    embeddings = model.embed_documents(texts)

    db = ChromaVectorStore(str(VECTOR_DIR), COLLECTION_NAME)
    db.add_chunks(file_chunks, embeddings)
    
    
if __name__ == "__main__":
    run_ingestion()