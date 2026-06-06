from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

class Splitter:
    def __init__(self):
        json_path = Path("JSON_LOGS/document_mapper.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.urls = json.load(f)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            # is_separator_regex=True,
            # separators=["\n", "Article", r"\(\d+\)"],
        )


    def get_url(self, file_path):
        for celex_id in self.urls.keys():
            if celex_id in file_path:
                return self.urls[celex_id]