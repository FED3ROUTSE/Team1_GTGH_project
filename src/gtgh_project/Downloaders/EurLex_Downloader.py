from pathlib import Path
from urllib.parse import urlparse
import requests
import json


class EurLexDownloader:
    def __init__(
        self,
        celex_id: str,
        file_type: str = "PDF",
        language: str = "EN",
        out_dir: str = "eu_docs",
    ):
        self.celex_id = celex_id
        self.language = language
        self.file_type = file_type.upper()
        self.out_dir = Path(out_dir)
        self.base_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/"

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._link_mapper()

    def _get_site_url(self) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _build_url(self) -> str:
        return (
            f"{self.base_url}{self.file_type}/"
            f"?uri=CELEX:{self.celex_id}&from={self.language}"
        )

    def _save_file(self, content: bytes) -> Path:
        file_path = self.get_file_path()
        file_path.write_bytes(content)
        return file_path

    def get_file_path(self) -> Path:
        file_type_end = self.file_type.lower()
        return self.out_dir / f"{self.celex_id}_{self.language}.{file_type_end}"

    def exists(self) -> bool:
        return self.get_file_path().exists()

    def download(self) -> Path:
        url = self._build_url()
        file_type_end = self.file_type.lower()

        headers = {
            "User-Agent": "Python EUR-Lex downloader/1.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if file_type_end not in content_type.lower():
            print(
                f"Warning: response may not be a supported file type. "
                f"Content-Type: {content_type}"
            )

        return self._save_file(response.content)

    def _link_mapper(self) -> Path:
        json_path = Path("JSON_LOGS/document_mapper.json")

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        data[self.celex_id] = self._get_site_url()

        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return json_path