"""작업 A — 번역 백엔드 통합 추상화.

translate(text, tgt_lang, backend) 형태로 감싸서 백엔드(DeepL / NLLB / glossary)를
교체 가능하게 한다. 배치 처리가 필요한 실사용은 translate_batch()를 쓴다.

백엔드:
    "deepl"     -> DeepLProvider (app/mt/deepl_provider.py)
    "nllb"      -> CTranslate2Provider (app/mt/ctranslate2_provider.py, kor_Hang -> target)
    "glossary"  -> 번역하지 않고 입력 텍스트를 그대로 반환 (loanword passthrough 등에 사용)
"""
from __future__ import annotations

import logging
from functools import lru_cache

from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.mt.ctranslate2_provider import get_default_provider
from app.mt.deepl_provider import DeepLKeyMissingError, DeepLProvider

logger = logging.getLogger("translate")


class BackendUnavailable(RuntimeError):
    """해당 백엔드로 번역할 수 없어 호출자가 다른 백엔드로 폴백해야 함을 알림."""


@lru_cache(maxsize=1)
def get_deepl_provider() -> DeepLProvider | None:
    """DeepL provider 싱글턴. 키가 없으면 None (매번 새로 만들지 않도록 캐시)."""
    try:
        return DeepLProvider()
    except DeepLKeyMissingError:
        logger.warning("DEEPL_API_KEY 미설정 -> DeepL 사용 불가, NLLB로 폴백합니다.")
        return None


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_not_exception_type(DeepLKeyMissingError),
    reraise=True,
)
def _deepl_translate_batch(texts: list[str], deepl_lang: str) -> list[str]:
    provider = get_deepl_provider()
    if provider is None:
        raise BackendUnavailable("DeepL 키가 없습니다.")
    return provider.translate_batch(texts, target_lang=deepl_lang)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def _nllb_translate_batch(texts: list[str], flores_lang: str) -> list[str]:
    provider = get_default_provider()
    return provider.translate_batch(texts, target_lang=flores_lang)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def back_translate_to_korean(texts: list[str], source_lang: str) -> list[str]:
    """target_lang 텍스트를 한국어로 역번역 (의심스러운 번역 탐지용, Phase 3.5).

    기존 translate_batch()는 항상 한국어를 소스로 가정하고 있어서(app/mt/lang_codes의
    SOURCE_LANG 고정) 역방향 호출이 안 된다. 이건 DeepL 쿼터를 아끼려고 항상 NLLB만
    쓴다 — 역번역은 "정답"이 아니라 의심 점수를 매기기 위한 참고용이라 품질 요구가
    낮고, 1298개 x 5개 언어라 호출량이 많다.
    """
    provider = get_default_provider()
    return provider.translate_batch(texts, target_lang="kor_Hang", source_lang=source_lang)


def translate_batch(
    texts: list[str],
    tgt_lang: str,
    backend: str,
    deepl_lang: str | None = None,
) -> list[str]:
    """texts 리스트를 한 번에 번역. tgt_lang은 FLORES-200 코드 (NLLB용)."""
    if not texts:
        return []

    if backend == "glossary":
        return list(texts)

    if backend == "deepl":
        if not deepl_lang:
            raise ValueError("backend='deepl' 이면 deepl_lang이 필요합니다.")
        return _deepl_translate_batch(texts, deepl_lang)

    if backend == "nllb":
        return _nllb_translate_batch(texts, tgt_lang)

    raise ValueError(f"알 수 없는 backend: {backend!r}")


def translate(text: str, tgt_lang: str, backend: str, deepl_lang: str | None = None) -> str:
    """단일 텍스트 번역. 배치가 필요하면 translate_batch를 직접 쓸 것."""
    return translate_batch([text], tgt_lang, backend, deepl_lang)[0]
