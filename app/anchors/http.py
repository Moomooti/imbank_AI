"""작업 B — 예의 바른(polite) HTTP 클라이언트.

- 식별 가능한 User-Agent (연락처 포함)
- 요청 간 고정 딜레이 (차단 방지)
- 일시적 오류(타임아웃/5xx)는 지수백오프로 재시도
"""
from __future__ import annotations

import logging
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("anchors.http")

USER_AGENT = "Mozilla/5.0 (compatible; FinHOLLY-research/0.1; +mailto:yues7864@gmail.com)"
REQUEST_DELAY_SEC = 1.5


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, **kwargs)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SEC)  # 매 요청마다 딜레이 (성공/실패 무관하게 서버 부담 줄임)
    return resp
