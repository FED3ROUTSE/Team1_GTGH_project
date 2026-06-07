from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.splitter_factory import SplitterFactory
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (EMBEDDING_MODEL_NAME, VECTOR_DIR,
COLLECTION_NAME)
import os 
import requests
from pathlib import Path

def run_ingestion():
    folder = "eu_docs"
    failed_downloads = []
    downloader = EurLexDownloader(file_type = "HTML")
    ROOT = Path(__file__).parent.parent.parent
    celex_list = [
        "32022R2554",
        "32022R0868",
        "32023R2854",
        "32022L2555",
        "32024R1689",
        "32019L1024",
        # "124124"
    ]

    for celex in celex_list:
        try:
            downloader.download(celex)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download file with celex_id: {celex}")
            failed_downloads.append(celex)

    splitter = SplitterFactory(downloader.file_type).get_splitter()

    file_chunks = []

    texts = []

    for file in os.listdir(folder):
        print(file)
        name, extension = file.split(".")
        if extension!=downloader.file_type.lower() or name.replace("_EN", "") not in celex_list:
            continue
        try:
            file_chunks += splitter.split(file_path = folder + "/" + file)
        except: 
            print(f"File {file} could not be split")
    
    texts.extend(chunk["content"] for chunk in file_chunks)
    
    model_name = EMBEDDING_MODEL_NAME
    model = LocalEmbeddingModel(model_name)
    embeddings = model.embed_documents(texts)

    persist_path = str(VECTOR_DIR)
    collection_name = COLLECTION_NAME

    db = ChromaVectorStore(persist_path, collection_name)
    db.add_chunks(file_chunks, embeddings)
    
    
if __name__ == "__main__":
    run_ingestion()