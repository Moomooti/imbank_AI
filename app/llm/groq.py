"""Phase 4-A — Groq REST 클라이언트. Gemini rate limit 폴백용.

(원래는 llama-3.3-70b-versatile를 쓰려 했으나 2026-09 기준 이 계정에서 사용
불가 -- Groq 계정에서 실제로 조회한 모델 목록 중 가장 범용적인 gpt-oss-120b로
대체.)

일일 토큰 한도(TPD)가 모델별로 별도 버킷이라, 대조데이터 생성 중(2026-09-10)
gpt-oss-120b 하나가 200,000 TPD를 다 써서 200건이 실패했다. 그래서 한 모델이
소진되면 즉시(재시도로 시간 버리지 않고) 다음 모델로 넘어가는
generate_with_model_fallback()을 추가함 -- 이게 사실상의 기본 진입점이다.

OpenAI 호환 chat/completions 스키마라 SDK 없이 requests로 충분하다.
"""
from __future__ import annotations

import os

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
# qwen/qwen3.6-27b는 제외함 -- 추론(thinking) 모델이라 번역 대신
# <think>...</think> 사고과정 텍스트를 통째로 뱉어서 번역 결과가 오염됨
# (2026-09-10 실측 확인, 실제로는 사용되기 전에 잡아서 오염 0건).
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_SEC = 60


class GroqUnavailable(RuntimeError):
    pass


class GroqRateLimited(RuntimeError):
    pass


def _get_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise GroqUnavailable(
            "GROQ_API_KEY가 .env에 없습니다. https://console.groq.com/keys 에서 발급 후 .env에 넣어주세요."
        )
    return key


def _call_once(prompt: str, model: str, temperature: float) -> str:
    """재시도 없이 한 번만 호출 (모델별 폴백 루프에서 쓰기 위함 -- 이미 소진된
    모델에 재시도 대기를 낭비하지 않도록)."""
    key = _get_key()
    headers = {"Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=TIMEOUT_SEC)
    except requests.RequestException as e:
        raise GroqUnavailable(f"요청 실패: {e!r}") from e

    if resp.status_code == 429:
        raise GroqRateLimited(f"rate limit: {resp.text[:300]}")
    if resp.status_code != 200:
        raise GroqUnavailable(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise GroqUnavailable(f"응답 파싱 실패: {data!r}") from e


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    retry=retry_if_exception_type(GroqRateLimited),
    reraise=True,
)
def generate(prompt: str, model: str = MODEL, temperature: float = 0.2) -> str:
    """단일 모델 호출 (기존 API 호환용). 모델 하나가 소진됐을 가능성이 있으면
    generate_with_model_fallback을 쓰는 게 낫다."""
    return _call_once(prompt, model, temperature)


def generate_with_model_fallback(prompt: str, temperature: float = 0.2, models: list[str] | None = None) -> tuple[str, str]:
    """FALLBACK_MODELS를 순서대로 즉시 시도(모델당 재시도 없이) -- 한 모델의
    일일 한도(TPD)가 다 차도 다른 모델 버킷으로 바로 넘어간다.
    (텍스트, 실제 사용 모델) 반환.
    """
    last_error: Exception | None = None
    for model in models or FALLBACK_MODELS:
        try:
            return _call_once(prompt, model, temperature), model
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    raise GroqUnavailable(f"모든 Groq 모델 실패: {last_error!r}")
