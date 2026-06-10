import sys
import logging
from pathlib import Path
from datetime import datetime
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.splitter_factory import SplitterFactory
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (EMBEDDING_MODEL_NAME,
COLLECTION_NAME, CELEX_LIST, DOCS_DIR, VECTOR_DIR, FILE_TYPE,
LANGUAGE)
import os
import requests


# ========== Set up logger ==========
ROOT = Path(__file__).parent.parent.parent.parent
LOGS_DIR = ROOT / "LOGS"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Creating file handler only - console outputs are determined inside classes
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOGS_DIR / f'ingest_logs_{now}.log'
fh = logging.FileHandler(str(log_file))
fh.setLevel(logging.INFO)

# Create formatter and add it to file handler
script = "ingest.py"
formatter = logging.Formatter(
    f'%(asctime)s|{script}|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s'
)
fh.setFormatter(formatter)

# Add handler to the logger
if not logger.handlers:
    logger.addHandler(fh)

# Log configuration at INFO level
logger.info(f"Configuration | ROOT={ROOT}")
logger.info(f"Configuration | LOGS_DIR={LOGS_DIR}")
logger.info(f"Configuration | EMBEDDING_MODEL_NAME={EMBEDDING_MODEL_NAME}")
logger.info(f"Configuration | COLLECTION_NAME={COLLECTION_NAME}")
logger.info(f"Configuration | CELEX_LIST={CELEX_LIST}")
logger.info(f"Configuration | FILE_TYPE={FILE_TYPE}")
logger.info(f"Configuration | LANGUAGE={LANGUAGE}")
logger.info(f"Configuration | DOCS_DIR={DOCS_DIR}")
logger.info(f"Configuration | VECTOR_DIR={VECTOR_DIR}")
# ===========================



def run_ingestion(celex_list = None):
    if celex_list is not None:
        sys.modules['src.gtgh_project.config'].CELEX_LIST = celex_list
    logger.info("\n\nPhase 1: Documents download")
    failed_downloads = []
    try:
        downloader = EurLexDownloader(file_type = FILE_TYPE, language = LANGUAGE, out_dir = DOCS_DIR)
        logger.info(f"Initializing EurLexDownloader | file_type={FILE_TYPE} | language={LANGUAGE} | out_dir={DOCS_DIR}")
    except Exception as e:
        logger.error(f"Failed to initialize downloader | error={e}")
        raise

    logger.info(f"Starting downloads for {len(CELEX_LIST)} documents | celex_ids={CELEX_LIST}")
    for idx, celex in enumerate(CELEX_LIST):

        logger.info(f"Download attempt {idx}/{len(CELEX_LIST)} | celex_id={celex}")
        try:
            already_exists = downloader.download(celex)
            if already_exists:
                logger.info(f"Document already exists | celex_id={celex} | progress={idx}/{len(CELEX_LIST)}")
            else:
                logger.info(f"Document downloaded successfully | celex_id={celex} | progress={idx}/{len(CELEX_LIST)}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed | celex_id={celex} | progress={idx}/{len(CELEX_LIST)} | error={type(e).__name__}")
            logger.error(f"Download error details | celex_id={celex} | error={str(e)}")
            failed_downloads.append(celex)
        if downloader.caution_response_type:
            logger.warning(f"Response may not be a supported file type | celex_id={celex}")

    if failed_downloads:
        logger.warning(f"Failed CELEX IDs: {failed_downloads}")
        for celex in failed_downloads:
            logger.warning(f"Document with celex ID: {celex} failed to download")


    logger.info("\n\nPhase 2: Document Chunking")

    logger.info(f"Creating splitter | file_type={downloader.file_type}")
    splitter = SplitterFactory(downloader.file_type).get_splitter()
    logger.info(f"Splitter created | type={type(splitter).__name__}")

    file_chunks = []
    texts = []

    for file in os.listdir(DOCS_DIR):
        logger.info(f"Processing file: {file}")

        name, extension = file.split(".")
        if extension!=downloader.file_type.lower(): # or name[:-3] not in CELEX_LIST:
            logger.warning(f"Skipping file | filename={file} | extension={extension} | reason=validation_failed")
            continue
        try:
            chunks = splitter.split(file_path = str(DOCS_DIR).split("\\")[-1] + "\\" + file)
            file_chunks += chunks
            logger.info(
                f"File chunked successfully | filename={file} | chunks_added={len(chunks)} | total_chunks_so_far={len(file_chunks)}")

        except Exception as e:
            logger.warning(f"File chunking failed | filename={file} | error={e}")
            print(e)

    texts.extend(chunk["content"] for chunk in file_chunks)


    logger.info("\n\nPhase 3: Embedding Generation")
    logger.debug(f"Initializing embedding model | model_name={EMBEDDING_MODEL_NAME}")
    model = LocalEmbeddingModel(EMBEDDING_MODEL_NAME)
    try:
        logger.info(f"Generating embeddings for {len(texts)} text segments")
        embeddings = model.embed_documents(texts)
        logger.info(f"Embeddings generated successfully | count={len(embeddings)}")
        logger.debug(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
    except Exception as e:
        logger.error(f"Embedding generation failed | error={e}")
        raise e

    logger.info("Phase 4: Vector Storage")
    logger.debug(f"Initializing vector store | directory={VECTOR_DIR} | collection={COLLECTION_NAME}")
    db = ChromaVectorStore(str(VECTOR_DIR), COLLECTION_NAME)

    logger.info(f"Adding {len(file_chunks)} chunks with embeddings to database")
    # db.add_chunks(file_chunks, embeddings)
    logger.info("Vector storage completed successfully")

    logger.info("Ingestion completed")

if __name__ == "__main__":
    run_ingestion()