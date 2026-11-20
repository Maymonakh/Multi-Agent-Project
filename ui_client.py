from __future__ import annotations

import json
import os
from urllib.parse import quote

import requests


class ApiClientError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code      # 0 = the API could not be reached
        self.detail = detail


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):            # FastAPI validation errors
        detail = "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in detail)
    return str(detail or response.text or f"HTTP {response.status_code}")


class ApiClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 180):
        self.base_url = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("API_KEY", "")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs):
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            response = requests.request(method, self.base_url + path, headers=headers,
                                        timeout=self.timeout, **kwargs)
        except requests.ConnectionError as error:
            raise ApiClientError(0, f"Cannot reach the API at {self.base_url}. Is it running?") from error
        except requests.Timeout as error:
            raise ApiClientError(0, "The API took too long to answer.") from error

        if response.status_code >= 400:
            raise ApiClientError(response.status_code, _detail(response))
        return None if response.status_code == 204 else response.json()

    # ---- system ----
    def health(self) -> dict:
        return self._request("GET", "/health")

    def clear_cache(self) -> None:
        self._request("DELETE", "/cache")

    # ---- documents ----
    def documents(self) -> list[str]:
        return self._request("GET", "/documents")["documents"]

    def upload_document(self, filename: str, data: bytes) -> dict:
        return self._request("POST", "/documents", files={"file": (filename, data)})

    def upload_image(self, filename: str, data: bytes) -> dict:
        return self._request("POST", "/documents/image", files={"image": (filename, data)})

    def delete_document(self, name: str) -> None:
        self._request("DELETE", f"/documents/{quote(name, safe='')}")

    # ---- OCR / chat ----
    def ocr(self, filename: str, data: bytes) -> dict:
        return self._request("POST", "/ocr", files={"image": (filename, data)})

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        return self._request("POST", "/ask", json={"question": question, "history": (history or [])[-20:]})

    def ask_image(self, filename: str, data: bytes, question: str = "", history: list[dict] | None = None) -> dict:
        return self._request(
            "POST", "/ask/image",
            files={"image": (filename, data)},
            data={"question": question, "history": json.dumps((history or [])[-20:])},
        )
