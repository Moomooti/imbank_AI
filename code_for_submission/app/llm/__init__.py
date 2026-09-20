"""Phase 4-A — LLM 클라이언트 통합 (Groq 우선).

원래 Gemini 우선 + Groq 폴백 구조였는데, 2026-09-10 Gemini가 하루 종일
503(고부하)에 계속 걸리면서 건마다 재시도 대기(최대 약 99초)만 날리는
문제가 발생. 이미 Groq가 실제 처리량의 92%를 담당하고 있었고 품질도
비슷했으므로(승인율 Gemini 71% vs Groq 68%, 표본 작음), Groq를 1순위로
바꾸고 Gemini는 완전히 뺐다 -- 나중에 Groq도 rate limit 걸리면 그때
재검토.
"""
from __future__ import annotations

import logging

from app.llm import groq

logger = logging.getLogger("llm")


class LLMUnavailable(RuntimeError):
    """Groq(전체 모델)가 다 실패했을 때."""


def generate(prompt: str, temperature: float = 0.2) -> tuple[str, str]:
    """(응답 텍스트, 실제 사용된 provider 이름) 반환. Groq 단독(모델 폴백 포함)."""
    text, model = groq.generate_with_model_fallback(prompt, temperature=temperature)
    return text, f"groq:{model}"


__all__ = ["generate", "LLMUnavailable"]
