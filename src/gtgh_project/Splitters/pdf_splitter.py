
# import necessary libraries
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256


class PdfSplitter:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.obj = self.load()


    def load(self):
        return fitz.open(self.pdf_path)


    def split(self, url=None):

        chunks_list = []

        # Iterate through pages correctly
        for page_num, page in enumerate(self.obj):  # self.obj is the fitz Document
            # Get page text
            text = page.get_text()

            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                is_separator_regex=True,
                separators=["\n", "Article", r"\(\d+\)"]
            )

            # Split page text into chunks
            chunks = text_splitter.split_text(text)

            # Create chunks for this page
            for chunk_index, chunk in enumerate(chunks):
                # Create chunk ID using hashlib - need to encode properly
                chunk_id_input = f"{self.pdf_path}|{page_num}|{chunk_index}|{chunk_index + 1}"
                chunk_id = sha256(chunk_id_input.encode("utf-8")).hexdigest()  # Add .hexdigest()

                chunks_list.append({
                    "chunk_id": chunk_id,
                    "source_file": self.pdf_path,
                    "page_number": page_num + 1,
                    "chunk_index": chunk_index,
                    "content": chunk,
                    "URL": url,
                    "date": self.obj.metadata["creationDate"],
                    "title": self.obj.metadata["title"]
                })

        return chunks_list



if __name__ == "__main__":
    # Usage
    folder = "../eu_docs"
    filepath = folder + "/32022L2555_EN.pdf"
    splitter = PdfSplitter(filepath)
    chunks = splitter.split(None)
    for chunk in chunks[20:25]:
        print(chunk["metadata"])
