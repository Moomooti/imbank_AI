"""작업 A — DeepL API provider.

CTranslate2Provider와 동일한 인터페이스(translate_batch)를 맞춰서
translate.py에서 백엔드를 교체 가능하게 한다.
"""
from __future__ import annotations

import os

import deepl


class DeepLKeyMissingError(RuntimeError):
    """DEEPL_API_KEY가 설정되지 않았을 때."""


class DeepLProvider:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("DEEPL_API_KEY", "").strip()
        if not api_key:
            raise DeepLKeyMissingError("DEEPL_API_KEY가 설정되어 있지 않습니다.")
        self.client = deepl.DeepLClient(api_key)

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "KO",
    ) -> list[str]:
        """target_lang은 DeepL 코드 (예: 'VI', 'ID', 'TH')."""
        results = self.client.translate_text(
            texts,
            source_lang=source_lang,
            target_lang=target_lang,
        )
                                                
        if not isinstance(results, list):
            results = [results]
        return [r.text for r in results]
