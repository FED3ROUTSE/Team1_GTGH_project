from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
from src.gtgh_project.Splitters.splitter_factory import SplitterFactory

import os
import requests 

if __name__ == "__main__":
    folder = "eu_docs"
    failed_downloads = []
    downloader = EurLexDownloader(file_type = "HTML")
    celex_list = [
        "32022R2554",
        "32022R0868",
        "32023R2854",
        "32022L2555",
        "32024R1689",
        "32019L1024",
        # "124124"
    ]

    for celex in celex_list:
        try:
            downloader.download(celex)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download file with celex_id: {celex}")
            failed_downloads.append(celex)

    splitter = SplitterFactory(downloader.file_type).get_splitter()

    file_chunks = []
    for file in os.listdir(folder):
        name, extension = file.split(".")
        if extension!=downloader.file_type.lower() or name not in celex_list:
            continue
        try:
            file_chunks += splitter.split(file_path = folder + "/" + file)
        except: 
            print(f"File {file} could not be split")