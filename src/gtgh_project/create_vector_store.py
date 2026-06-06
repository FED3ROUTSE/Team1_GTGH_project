
from src.gtgh_project.Splitters.pdf_splitter import PdfSplitter
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from pathlib import Path
import os
from src.gtgh_project.ChromaDB.vector_store import ChromaVectorStore

ROOT = Path(__file__).parent.parent.parent
pdf_path = ROOT / "eu_docs"
docs = os.listdir(pdf_path)

splitter = PdfSplitter()

all_chunks = []         # keep for DB
texts = []

for doc in docs:
    pdf_chunks = splitter.split_pdf(str(pdf_path / Path(doc)))
    all_chunks.extend(pdf_chunks)
    texts.extend(chunk["content"] for chunk in pdf_chunks)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = LocalEmbeddingModel(model_name)
embeddings = model.embed_documents(texts)

persist_path = str(ROOT / Path("Vector DB Folder"))         # Folder with table and metadata under root
collection_name = "vector_db"

db = ChromaVectorStore(persist_path, collection_name)
db.add_chunks(all_chunks, embeddings)