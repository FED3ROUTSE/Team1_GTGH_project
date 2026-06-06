from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.pdf_splitter import PdfSplitter
import os
import requests 

if __name__ == "__main__":
    failed_downloads = []
    pdf_downloader = EurLexDownloader()
    celex_list = [
        "32022R2554",
        "32022R0868",
        "32023R2854",
        "32022L2555",
        "32024R1689",
        "32019L1024",
    ]

    for celex in celex_list:
        try:
            pdf_downloader.download(celex)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download file with celex_id: {celex}")
            failed_downloads.append(celex)
    folder = "eu_docs"
    pdf_splitter = PdfSplitter()
    for file in os.listdir(folder):
        try:
            pdf_chunks = pdf_splitter.split_pdf(folder + "/" + file)
            print(pdf_chunks[0]["content"])
            print("-"*30)
        except: 
            print(f"File {file} could not be split")
