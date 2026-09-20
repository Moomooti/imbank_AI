"""Phase 4-A — Gemini 2.5 Flash (Google AI Studio 무료 티어) REST 클라이언트.

방금 unbabel-comet/sentence-transformers 버전 충돌을 겪었기 때문에, 무거운
google-generativeai SDK(자체 protobuf/grpc 의존성 트리) 대신 requests로
REST API를 직접 호출한다 -- 이 프로젝트는 seed 제작용 1회성 도구라 SDK의
편의 기능(스트리밍 등)이 필요 없다.

이 모듈이 호출하는 API는 예문/용어 같은 일반 금융 용어 데이터만 보낸다
(고객 데이터 아님 -- 온프레미스 원칙 유지, Phase 1 원칙과 동일).
"""
from __future__ import annotations

import os

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MODEL = "gemini-3.6-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_SEC = 120


class GeminiUnavailable(RuntimeError):
    """키가 없거나 API 호출이 최종적으로 실패했을 때."""


class GeminiRateLimited(RuntimeError):
    """429 -- 폴백(Groq)으로 넘어가라는 신호. 재시도로도 못 넘기면 호출자가 잡는다."""


class GeminiTransientError(RuntimeError):
    """네트워크 타임아웃 등 -- 재시도하면 될 수도 있는 일시적 오류."""


def _get_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiUnavailable(
            "GEMINI_API_KEY가 .env에 없습니다. https://aistudio.google.com/apikey 에서 발급 후 .env에 넣어주세요."
        )
    return key


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    retry=retry_if_exception_type((GeminiRateLimited, GeminiTransientError)),
    reraise=True,
)
def generate(prompt: str, model: str = MODEL, temperature: float = 0.2) -> str:
    """단일 프롬프트 -> 텍스트 응답. 429(rate limit)·타임아웃은 지수백오프로 재시도."""
    key = _get_key()
    url = ENDPOINT.format(model=model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, params={"key": key}, json=body, timeout=TIMEOUT_SEC)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise GeminiTransientError(f"네트워크 일시 오류: {e!r}") from e
    except requests.RequestException as e:
        raise GeminiUnavailable(f"요청 실패: {e!r}") from e

    if resp.status_code == 429:
        raise GeminiRateLimited(f"rate limit: {resp.text[:300]}")
    if resp.status_code != 200:
        raise GeminiUnavailable(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        candidates = data["candidates"]
        if not candidates:
            raise KeyError("candidates 비어있음")
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise GeminiUnavailable(f"응답 파싱 실패: {data!r}") from e
