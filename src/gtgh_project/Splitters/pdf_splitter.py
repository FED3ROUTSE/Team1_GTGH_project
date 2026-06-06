# import necessary libraries
import pymupdf
from hashlib import sha256
from src.gtgh_project.Splitters.splitter import Splitter

class PdfSplitter(Splitter):
    def __init__(self):
        super().__init__()

    def split(self, file_path):
        pdf_file = pymupdf.open(file_path)
        url = self.get_url(file_path)
        chunks_list = []

        for page_num, page in enumerate(pdf_file):

            text = page.get_text()
            chunks = self.text_splitter.split_text(text)

            # Create chunks for this page
            for chunk_index, chunk in enumerate(chunks):
                chunk_id_input = (
                    f"{file_path}|{page_num}|{chunk_index}|{chunk_index + 1}"
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
