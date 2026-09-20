"""작업 B — 한/영 정기 통계자료 짝짓기.

시점(period)만으로는 부족하다는 게 실제로 확인됐다: 같은 달에 자본비율/부실채권/
가계대출 등 여러 통계자료가 동시에 나와서, 시점만 같으면 전혀 다른 주제끼리
잘못 묶이는 사고가 발생했다. 그래서 매칭 키를 (series_id, period) 로 잡는다 —
"이 문서가 어떤 통계 시리즈인지"까지 확인한 다음에만 같은 시점끼리 짝짓는다.

series_id를 못 찾은 항목(=우리가 아직 모르는 시리즈)은 애초에 통계자료 후보에서
제외된다(annotate_and_filter). 같은 (series_id, period)에 후보가 여럿 남는 극히
드문 경우에만 제목을 NLLB로 초벌 번역해 유사도로 최종 타이브레이크하되, 이미
같은 시리즈로 확인된 후보들 사이의 타이브레이크라 오매칭 위험이 훨씬 낮다.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Callable

FUZZY_TIEBREAK_THRESHOLD = 0.6


def annotate_and_filter(items: list[dict], lang: str, parse_period) -> list[dict]:
    """items에 period + series 키를 붙이고, 둘 다 식별된 것만 남긴다."""
    from app.anchors.period import is_stat_title
    from app.anchors.series import match_series

    out = []
    for it in items:
        if lang == "ko":
            fallback_year = int(it["date"][:4]) if it.get("date") else None
            period = parse_period(it["title"], fallback_year)
        else:
            period = parse_period(it["title"])
        series_id = match_series(it["title"], lang)
        if is_stat_title(period, it["title"]) and series_id:
            it["period"] = period
            it["series"] = series_id
            out.append(it)
    return out


def build_pairs(
    ko_items: list[dict],
    en_items: list[dict],
    translate_title_fn: Callable[[list[str]], list[str]],
) -> tuple[list[tuple[dict, dict, str]], list[dict]]:
    ko_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    en_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in ko_items:
        ko_by_key[(it["series"], it["period"])].append(it)
    for it in en_items:
        en_by_key[(it["series"], it["period"])].append(it)

    pairs: list[tuple[dict, dict, str]] = []
    unmatched: list[dict] = []

    for key, ko_cands in ko_by_key.items():
        en_cands = en_by_key.get(key, [])
        if not en_cands:
            unmatched.extend({"side": "ko", "period": key[1], "series": key[0], **c} for c in ko_cands)
            continue

        if len(ko_cands) == 1 and len(en_cands) == 1:
            pairs.append((ko_cands[0], en_cands[0], f"series+period:{key[0]}"))
            continue

        # 같은 시리즈+시점에 후보가 여럿인 드문 경우만 제목 유사도로 타이브레이크
        translated = dict(zip((c["title"] for c in ko_cands), translate_title_fn([c["title"] for c in ko_cands])))
        used_en_ids = set()
        for kc in ko_cands:
            kc_en = translated[kc["title"]].lower()
            best, best_score = None, 0.0
            for ec in en_cands:
                if id(ec) in used_en_ids:
                    continue
                score = SequenceMatcher(None, kc_en, ec["title"].lower()).ratio()
                if score > best_score:
                    best, best_score = ec, score
            if best is not None and best_score >= FUZZY_TIEBREAK_THRESHOLD:
                pairs.append((kc, best, f"series+fuzzy:{key[0]}:{best_score:.2f}"))
                used_en_ids.add(id(best))
            else:
                unmatched.append({"side": "ko", "period": key[1], "series": key[0], **kc})

    for key, en_cands in en_by_key.items():
        if key not in ko_by_key:
            unmatched.extend({"side": "en", "period": key[1], "series": key[0], **c} for c in en_cands)

    return pairs, unmatched
