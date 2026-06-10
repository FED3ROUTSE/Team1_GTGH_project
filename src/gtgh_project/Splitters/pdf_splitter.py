import pymupdf
from hashlib import sha256
from src.gtgh_project.Splitters.splitter import Splitter
import re

class PdfSplitter(Splitter):
    def __init__(self):
        super().__init__()

    
    @staticmethod
    def extract_pdf_metadata(text: str):
        
        # Find the date with its surrounding context
        date_pattern = r'of\s+\d{1,2}\s+\w+\s+\d{4}\s*\n'
        date_match = re.search(date_pattern, text)

        if not date_match:
            return None, None

        # title starts after the date line
        title_start = date_match.end()

        # Look for the end pattern (case insensitive)
        end_pattern = r'THE EUROPEAN PARLIAMENT AND THE COUNCIL'
        end_match = re.search(end_pattern, text, re.IGNORECASE)

        if not end_match:
            return None, None

        title_end = end_match.start()

        # extract the title
        title = text[title_start:title_end].strip()

        # replace newlines and multiple spaces with single space
        title = re.sub(r'\s+', ' ', title)

        # remove any trailing " (Text with EEA relevance)" if present
        title = re.sub(r'\s*\(Text with EEA relevance\)\s*$', '', title)

        return title, date_match.group()

    def split(self, file_path):
        pdf_file = pymupdf.open(file_path)
        url = self.get_url(file_path)
        chunks_list = []
        pdf_title, pdf_date = self.extract_pdf_metadata(pdf_file[0].get_text())
        if pdf_title is None:
            pdf_title = "Unknown"
            pdf_date = "Unknown"
        
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
                        "date": pdf_date,
                        "title": pdf_title,
                    }
                )

        return chunks_list
