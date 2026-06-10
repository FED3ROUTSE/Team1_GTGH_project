from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.LLMs.rag_chain import RagChain
from src.gtgh_project.LLMs.retriever import Retriever
import logging
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (VECTOR_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME,
TOP_K, LOCAL, TEMPERATURE)
from pathlib import Path
from datetime import datetime


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


app = FastAPI()

# enable cors for all origins
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")

logger.info("Initializing Embedding Model")
embedding_model = LocalEmbeddingModel(
    model_name=EMBEDDING_MODEL_NAME,
)
logger.info(f"Embedding Model Loaded | model={EMBEDDING_MODEL_NAME}")

logger.info("Initializing Vector Store")
vector_store = ChromaVectorStore(
    persist_path=VECTOR_DIR,
    collection_name=COLLECTION_NAME,
)
logger.info(f"Vector Store Initialized | path={VECTOR_DIR} | collection={COLLECTION_NAME}")

logger.info("Initializing RAG Chain")
rag_chain = RagChain(local_llm=LOCAL, temperature=TEMPERATURE)
logger.info(f"RAG Chain Initialized | local_llm={LOCAL} | temperature={TEMPERATURE}")

logger.info("Initializing RAG Engine")
rag_engine = Retriever(embedding_model=embedding_model, vector_store=vector_store, llm = rag_chain, top_k = TOP_K, fetch_k=20, lambda_mult=0.6)
logger.info("RAG Engine Initialized Successfully")
logger.info(f"RAG Engine LLM={rag_engine.llm}")

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list


@app.post("/query")
def query(request: QueryRequest):
    logger.info(f"Query Received | question_length={len(request.question)}")
    logger.info(f"Question Content: {request.question[:200]}...")

    try:
        logger.info("Calling rag_engine.ask")
        result = rag_engine.ask(request.question)
        logger.info(
            f"Query successful | answer_length={len(result.get('answer', ''))} | chunks_retrieved={len(result.get('retrieved_chunks', []))}")
        logger.debug(f"Answer preview: {result.get('answer', '')[:200]}...")
        return result
    except Exception as e:
        logger.error(f"Query Failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    logger.info("Serving Frontend Page")
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        logger.info(f"Frontend Served | path={html_path}")
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    logger.warning(f"Frontend not found | attempted_path={html_path}")
    return HTMLResponse(content="<h1>Frontend not found. Please ensure index.html is in the same directory.</h1>",
                        status_code=404)

logger.info("API application started successfully")