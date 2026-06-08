
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent.parent
LOGS_DIR = ROOT / "LOGS"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Creating file handler only - console outputs are determined inside classes
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOGS_DIR / f'ingest_logs_{now}.log'
fh = logging.FileHandler(str(log_file))
fh.setLevel(logging.INFO)

# Create formatter and add it to file handler
formatter = logging.Formatter(
    '%(asctime)s|%(name)s|%(levelname)s|%(message)s'
)
fh.setFormatter(formatter)

# Add handler to the logger
if not logger.handlers:
    logger.addHandler(fh)

logger.info("Ingestion started")
logger.info("")

from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.splitter_factory import SplitterFactory
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (EMBEDDING_MODEL_NAME,
COLLECTION_NAME, CELEX_LIST, DOCS_DIR, VECTOR_DIR, FILE_TYPE,
LANGUAGE)
import os 
import requests


def run_ingestion():

    logger.info("Document download started")
    failed_downloads = []
    try:
        downloader = EurLexDownloader(file_type = FILE_TYPE, language = LANGUAGE, out_dir = DOCS_DIR)
    except Exception as e:
        logger.error(e)
        raise

    for celex in CELEX_LIST:
        try:
            downloader.download(celex)
            logger.info(f"Document downloaded | celex_id={celex}")
        except requests.exceptions.RequestException as e:
            logger.error(e)
            print(f"Download failed | celex_id={celex}")
            failed_downloads.append(celex)

    if failed_downloads:
        for celex in failed_downloads:
            logger.warning(f"Document with celex ID: {celex} failed to download")

    logger.info("Document download completed")
    logger.info("")

    logger.info("Document chunking started")
    splitter = SplitterFactory(downloader.file_type).get_splitter()

    file_chunks = []
    texts = []

    for file in os.listdir(DOCS_DIR):
        name, extension = file.split(".")
        if extension!=downloader.file_type.lower() or name[:-3] not in CELEX_LIST:
            logger.warning(f"Skipping file | filename={file} | extension={extension} | reason=validation_failed")
            continue
        try:
            file_chunks += splitter.split(file_path = str(DOCS_DIR).split("\\")[-1] + "\\" + file)
            logger.info(f"File {file} chunked | Total chunks={len(file_chunks)}")
        except Exception as e:
            logger.warning(f"File chunking failed | filename={file} | error={e}")
            print(f"File {file} could not be split")
    
    texts.extend(chunk["content"] for chunk in file_chunks)

    logger.info("Document chunking completed")
    logger.info("")

    logger.info("Embedding generation started")
    model = LocalEmbeddingModel(EMBEDDING_MODEL_NAME)
    try:
        embeddings = model.embed_documents(texts)
    except Exception as e:
        logger.error(e)
        raise e

    db = ChromaVectorStore(str(VECTOR_DIR), COLLECTION_NAME)
    db.add_chunks(file_chunks, embeddings)
    logger.info("Embedding generation completed")
    logger.info("")
    
    logger.info("Ingestion completed")

if __name__ == "__main__":
    run_ingestion()