
# TEST FILE
# Test embedding process without retrying to download docs
# works

from src.gtgh_project.Splitters.pdf_splitter import PdfSplitter
from src.gtgh_project.Vectorization.embeddings import LocalEmbeddingModel
from pathlib import Path
import os


ROOT = Path(__file__).parent.parent.parent
pdf_path = ROOT / "eu_docs"
docs = os.listdir(pdf_path)

splitter = PdfSplitter()

texts = list()
for doc in docs:
    pdf_chunks = splitter.split_pdf(str(pdf_path / Path(doc)))      # Convert path to str to pass splitting "validation"
    texts.extend(chunk["content"] for chunk in pdf_chunks)


model_name = "see tutor's repo"
model = LocalEmbeddingModel(model_name)
embeddings = model.embed_documents(texts)
print(embeddings[:5])