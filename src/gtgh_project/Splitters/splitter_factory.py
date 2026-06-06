from src.gtgh_project.Splitters.pdf_splitter import PdfSplitter
from src.gtgh_project.Splitters.html_splitter import HTMLSplitter
from src.gtgh_project.Splitters.text_splitter import TEXTSplitter


class SplitterFactory:
    def __init__(self, file_type):
        self.splitter = self.__create_splitter(file_type)

    def __create_splitter(self, file_type):
        match file_type:
            case "PDF":
                return PdfSplitter()
            case "HTML":
                return HTMLSplitter()
            case "TXT":
                return TEXTSplitter()
            case _:
                return None

    
    def get_splitter(self):
        return self.splitter