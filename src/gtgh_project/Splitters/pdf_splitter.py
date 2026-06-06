# import necessary libraries
import json
from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256


class PdfSplitter:
    def __init__(self):
        json_path = Path("JSON_LOGS/document_mapper.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.urls = json.load(f)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            is_separator_regex=True,
            separators=["\n", "Article", r"\(\d+\)"],
        )

    def get_url(self, pdf_path):
        for celex_id in self.urls.keys():
            if celex_id in pdf_path:
                return self.urls[celex_id]

    def split_pdf(self, pdf_path):
        pdf_file = pymupdf.open(pdf_path)
        url = self.get_url(pdf_path)
        chunks_list = []

        for page_num, page in enumerate(pdf_file):

            text = page.get_text()
            chunks = self.text_splitter.split_text(text)

            # Create chunks for this page
            for chunk_index, chunk in enumerate(chunks):
                chunk_id_input = (
                    f"{pdf_path}|{page_num}|{chunk_index}|{chunk_index + 1}"
                )
                chunk_id = sha256(chunk_id_input.encode("utf-8")).hexdigest()

                chunks_list.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page_num + 1,
                        "chunk_index": chunk_index,
                        "content": chunk,
                        "URL": url,
                        "date": pdf_file.metadata["creationDate"],
                        "title": pdf_file.metadata["title"],
                    }
                )

        return chunks_list
