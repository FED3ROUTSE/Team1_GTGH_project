import html2text
from src.gtgh_project.Splitters.text_splitter import TEXTSplitter

class HTMLSplitter(TEXTSplitter):
    def __init__(self):
        super().__init__()

    
    def split(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        text = html2text.html2text(html_content)
        return super().split(file_path=file_path, text=text)
        