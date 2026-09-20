"""작업 B — 제목에서 '기준월/기준반기' 추출 + 정기 통계자료 판별.

목표: "Household Loans, July 2026" / "가계대출 동향(2026년 7월중)" 같은
정기 통계성 보도자료를 서사형 정책발표와 구분하고, 같은 시점을 가리키는
한/영 항목을 짝지을 때 쓸 공통 키(period key)를 뽑아낸다.
"""
from __future__ import annotations

import re

_EN_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
    )
}

_EN_MONTH_RE = re.compile(
    r"\b(" + "|".join(_EN_MONTHS) + r")\.?\s+(\d{4})\b", re.IGNORECASE
)
_EN_HALF_RE = re.compile(r"\b(first|second)\s+half\s+(\d{4})\b", re.IGNORECASE)
_EN_QUARTER_RE = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)

_KO_MONTH_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
# 금융당국 특유의 2자리 연도 축약형: "26.6말", "26.6월말", "26.6월중" 등
# (단, "26.9.10.부터" 같은 전체 날짜(YY.M.D)는 제외 -> 뒤에 말/중이 바로 와야 함)
_KO_MONTH_YY_RE = re.compile(r"(?<!\d)(\d{2})\.(\d{1,2})\s*월?\s*(?:말|중)")
_KO_MONTH_NOYEAR_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월(?:말|중)?")
_KO_HALF_RE = re.compile(r"(\d{4})\s*년\s*(상|하)반기")


def _period_repr(year: int, unit: str | int) -> str:
    """(2026, 6) -> '2026-06', (2026, 'H1') -> '2026-H1'"""
    if isinstance(unit, int):
        return f"{year}-{unit:02d}"
    return f"{year}-{unit}"


def parse_period_en(title: str) -> str | None:
    if m := _EN_HALF_RE.search(title):
        half = "H1" if m.group(1).lower() == "first" else "H2"
        return _period_repr(int(m.group(2)), half)
    if m := _EN_QUARTER_RE.search(title):
        return _period_repr(int(m.group(2)), f"Q{m.group(1)}")
    if m := _EN_MONTH_RE.search(title):
        return _period_repr(int(m.group(2)), _EN_MONTHS[m.group(1).lower()])
    return None


def parse_period_ko(title: str, fallback_year: int | None) -> str | None:
    """fallback_year: 제목에 연도가 없을 때 쓸 연도. None이면 연도 없는 제목은 버림
    (엉뚱한 연도로 잘못 매칭되는 것을 막기 위함)."""
    if m := _KO_HALF_RE.search(title):
        half = "H1" if m.group(2) == "상" else "H2"
        return _period_repr(int(m.group(1)), half)
    if m := _KO_MONTH_RE.search(title):
        return _period_repr(int(m.group(1)), int(m.group(2)))
    if m := _KO_MONTH_YY_RE.search(title):
        return _period_repr(2000 + int(m.group(1)), int(m.group(2)))
    if fallback_year is not None and (m := _KO_MONTH_NOYEAR_RE.search(title)):
        month = int(m.group(1))
        if not (1 <= month <= 12):
            return None
        return _period_repr(fallback_year, month)
    return None


def is_stat_title(period_key: str | None, title: str) -> bool:
    """정기 통계자료로 볼지 판별. period_key가 있고, 제목이 서사형치고 너무 길지 않을 때만."""
    return period_key is not None and len(title) <= 80
