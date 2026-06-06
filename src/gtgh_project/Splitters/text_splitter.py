from langchain_community.document_loaders import TextLoader
from hashlib import sha256
from src.gtgh_project.Splitters.splitter import Splitter

class TEXTSplitter(Splitter):

    def __init__(self):
        super().__init__()

    def split(self, file_path, text: str = None):
        if text is None:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        url = self.get_url(file_path)
        chunks_list = []
        chunks = self.text_splitter.split_text(text)
        for chunk_index, chunk in enumerate(chunks):
                chunk_id_input = (
                    f"{file_path}|{chunk_index}|{chunk_index + 1}"
                )
                chunk_id = sha256(chunk_id_input.encode("utf-8")).hexdigest()
                chunks_list.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": "Unknown",
                        "chunk_index": chunk_index,
                        "content": chunk,
                        "URL": url,
                        "date": "Unknown",
                        "title": "Unknown",
                    }
                )
        return chunks_list

        