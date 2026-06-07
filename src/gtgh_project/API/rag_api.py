
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from src.gtgh_project.RAGPipeline.rag import RAG
# import logging
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore
from src.gtgh_project.config import (
    DOCS_DIR,
    VECTOR_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from pathlib import Path

# logging.basicConfig(level=logging.INFO)

app = FastAPI()

# enable cors for all origins
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = LocalEmbeddingModel(
    model_name=EMBEDDING_MODEL_NAME,
)
# logging.info("--------------------")
# logging.info(CHROMA_PATH)
# logging.info("--------------------")

vector_store = ChromaVectorStore(
    persist_path=VECTOR_DIR,
    collection_name=COLLECTION_NAME,
)

rag = RAG()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list


@app.post("/query")
def query(request: QueryRequest):
    try:
        answer_text = rag.query(request.question)

        retrieved_chunks = []
        try:
            query_embedding = rag.embedding_model.embed_query(request.question)
            results = rag.vector_store.search(query_embedding, top_k=5)

            for result in results:
                retrieved_chunks.append({
                    "content": result['content'],
                    "metadata": result.get('metadata', {})
                })
        except:
            pass

        return {
            "question": request.question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Please ensure index.html is in the same directory.</h1>",
                        status_code=404)