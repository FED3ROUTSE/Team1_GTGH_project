from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.pdf_splitter import PdfSplitter
import os

if __name__ == "__main__":
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
        pdf_downloader.download(celex)

    folder = "eu_docs"
    pdf_splitter = PdfSplitter()
    for file in os.listdir(folder):
        pdf_chunks = pdf_splitter.split_pdf(folder + "/" + file)
        print(len(pdf_chunks))
