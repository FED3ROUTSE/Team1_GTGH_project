from src.gtgh_project.Downloaders.EurLex_Downloader import EurLexDownloader
if __name__ == "__main__":
    pdf_downloader = EurLexDownloader()
    celex_list = ["32022R2554", "32022R0868", "32023R2854", "32022L2555", "32024R1689", "32019L1024"]
    
    for celex in celex_list:
        pdf_downloader.download(celex)