from pathlib import Path
from urllib.parse import urlparse
import requests
import json


class EurLexDownloader:
    def __init__(self, file_type: str, language: str, out_dir: Path):
        self.language = language
        self.file_type = file_type
        self.out_dir = out_dir
        self.base_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/"

        self.out_dir.mkdir(parents=True, exist_ok=True)


    def _get_site_url(self) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _build_url(self, celex: str) -> str:
        return (
            f"{self.base_url}{self.file_type}/"
            f"?uri=CELEX:{celex}&from={self.language}"
        )

    def _save_file(self, content: bytes, celex: str):
        file_path = self.get_file_path(celex)
        file_path.write_bytes(content)

    def get_file_path(self, celex: str) -> Path:
        file_type_end = self.file_type.lower()
        return self.out_dir / f"{celex}_{self.language}.{file_type_end}"

    def exists(self, celex) -> bool:
        return self.get_file_path(celex).exists()

    def download(self, celex):
        if self.exists(celex):
            print(f"File {celex}_{self.language}.{self.file_type.lower()} already exists. It will not be downloaded again.")
            return
        url = self._build_url(celex)
        file_type_end = self.file_type.lower()

        headers = {
            "User-Agent": "Python EUR-Lex downloader/1.0"
        }
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error")
            print(errh.args[0])
            raise errh
        except requests.exceptions.ReadTimeout as errrt:
            print("Time out")
            raise errrt
        except requests.exceptions.ConnectionError as conerr:
            print("Connection error")
            raise conerr
        except requests.exceptions.RequestException as errex:
            print("Exception request")
            raise errex
        
        content_type = response.headers.get("Content-Type", "")

        if file_type_end not in content_type.lower():
            print(
                f"Warning: response may not be a supported file type. "
                f"Content-Type: {content_type}"
            )
        self._save_file(response.content, celex)
        self._link_mapper(celex)
        

    def _link_mapper(self, celex):
        json_path = Path("JSON_LOGS/document_mapper.json")

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        data[celex] = self._build_url(celex)

        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)