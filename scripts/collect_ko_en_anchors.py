"""Phase 1 - 작업 B: 금융위/금감원의 한/영 병기 "정기 통계자료" 보도자료를 짝지어
ko_en_anchors.csv 로 저장한다.

작업 A(translate_glossary.py)와 완전히 독립된 스크립트. NLLB는 오직 한글 제목을
영문으로 초벌 번역해 같은 시점에 여러 후보가 있을 때 제목 유사도로 짝짓는 데만
쓰인다 (본문 자체는 100% 기관이 직접 공개한 원문/공식번역).

■ 수집 범위 (2026-09-09 확인)
    - 금융위원회(FSC), 금융감독원(FSS): 목록·상세 모두 서버렌더링, robots.txt 제약 없음
    - 한국은행(BOK): 목록이 JS/AJAX 렌더링이라 보류 (app/anchors/config.py 참고)

■ 왜 "정기 통계자료"만 우선하나
    서사형 정책발표(예: 토큰증권 정책방향)는 한/영판 대조 결과 문장 대 문장 번역이
    아니라 영문판이 훨씬 길게 재구성된 별도 설명문이었다. 반면 "가계대출 동향",
    "은행 자기자본비율" 같은 정기 통계자료는 수치와 표현이 거의 1:1로 일치해서
    앵커 문장쌍으로 훨씬 적합하다.

■ 산출물
    ko_en_anchors.csv       (컬럼: ko, en, source, url + en_url/date/match_method)
    anchors_review_needed.csv  (자동 짝짓기 실패 후보 - 사람이 검토)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.anchors.config import SOURCES  # noqa: E402
from app.anchors.http import get  # noqa: E402
from app.anchors.matcher import annotate_and_filter, build_pairs  # noqa: E402
from app.anchors.parsers import PARSERS  # noqa: E402
from app.anchors.period import parse_period_en, parse_period_ko  # noqa: E402
from app.mt.translate import translate_batch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_ko_en_anchors")

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "anchors" / "ko_en_anchors.csv"
REVIEW_CSV = ROOT / "data" / "anchors" / "anchors_review_needed.csv"

FIELDNAMES = ["ko", "en", "source", "url", "en_url", "date", "match_method"]
REVIEW_FIELDNAMES = ["source", "side", "series", "period", "title", "url", "date"]


def translate_titles_to_en(titles: list[str]) -> list[str]:
    if not titles:
        return []
    return translate_batch(titles, tgt_lang="eng_Latn", backend="nllb")


def crawl_list(url_template: str, base_url: str, lang: str, parser_key: str, pages: int) -> list[dict]:
    """페이지를 순회하며 목록 항목을 모은다.

    실제 수집 중 특정 페이지가 일시적으로(추정: 서버 blip) 빈 결과를 준 적이
    있었다(재현은 안 됨). 그 한 번 때문에 이후 모든 페이지를 건너뛰면 안 되므로,
    빈 페이지 1번은 재시도하고, 그래도 비어 있으면 그 페이지만 건너뛴 채 계속
    진행한다. 연속으로 2번 비면 그때 목록이 진짜 끝난 것으로 보고 멈춘다.
    """
    parse_list = PARSERS[parser_key]["list"]
    items, seen_urls = [], set()
    consecutive_empty = 0
    for page in range(1, pages + 1):
        url = url_template.format(page=page)
        page_items: list[dict] = []
        for attempt in range(2):  # 빈 결과면 한 번 더 시도
            try:
                resp = get(url)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"목록 요청 실패 {url}: {e!r}")
                break
            page_items = parse_list(resp.text, base_url, lang)
            if page_items or attempt == 1:
                break
            logger.warning(f"  page {page}: 빈 결과, 재시도")

        new = [it for it in page_items if it["url"] not in seen_urls]
        for it in new:
            seen_urls.add(it["url"])
        items.extend(new)
        logger.info(f"  page {page}/{pages}: {len(new)}건 (누적 {len(items)})")

        consecutive_empty = consecutive_empty + 1 if not page_items else 0
        if consecutive_empty >= 2:
            logger.info(f"  page {page}: 연속 빈 페이지 -> 목록 끝으로 판단, 중단")
            break
    return items


def fetch_detail(url: str, parser_key: str) -> dict | None:
    try:
        resp = get(url)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"상세 요청 실패 {url}: {e!r}")
        return None
    return PARSERS[parser_key]["detail"](resp.text)


def load_existing_urls() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    import pandas as pd

    return set(pd.read_csv(OUT_CSV, encoding="utf-8-sig")["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=5, help="기관·언어별 크롤링할 목록 페이지 수")
    parser.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    args = parser.parse_args()

    existing_urls = load_existing_urls()
    out_rows: list[dict] = []
    review_rows: list[dict] = []

    sources = [k for k, v in SOURCES.items() if v["enabled"]] if args.source == "all" else [args.source]

    for key in sources:
        cfg = SOURCES[key]
        if not cfg["enabled"]:
            logger.info(f"[{key}] 비활성화({cfg.get('note', '')}) -> 스킵")
            continue

        logger.info(f"=== {cfg['name']} ({key}) ===")
        logger.info(" 한글 목록 수집...")
        ko_items = crawl_list(cfg["ko_list"], cfg["base_url"], "ko", cfg["parser"], args.pages)
        logger.info(" 영문 목록 수집...")
        en_items = crawl_list(cfg["en_list"], cfg["base_url"], "en", cfg["parser"], args.pages)

        ko_stat = annotate_and_filter(ko_items, "ko", parse_period_ko)
        en_stat = annotate_and_filter(en_items, "en", parse_period_en)
        logger.info(f" 통계성 제목 필터링: ko {len(ko_items)}->{len(ko_stat)}, en {len(en_items)}->{len(en_stat)}")

        pairs, unmatched = build_pairs(ko_stat, en_stat, translate_titles_to_en)
        logger.info(f" 짝짓기 결과: 확정 {len(pairs)}쌍, 미확정 {len(unmatched)}건")

        for ko_item, en_item, method in pairs:
            if ko_item["url"] in existing_urls:
                continue
            ko_detail = fetch_detail(ko_item["url"], cfg["parser"])
            en_detail = fetch_detail(en_item["url"], cfg["parser"])
            if not ko_detail or not en_detail or not ko_detail["body"] or not en_detail["body"]:
                logger.warning(f" 본문 추출 실패, 스킵: {ko_item['url']}")
                continue
            out_rows.append({
                "ko": ko_detail["body"],
                "en": en_detail["body"],
                "source": cfg["name"],
                "url": ko_item["url"],
                "en_url": en_item["url"],
                "date": ko_detail["date"] or ko_item.get("date") or "",
                "match_method": method,
            })
            existing_urls.add(ko_item["url"])

        for u in unmatched:
            review_rows.append({
                "source": cfg["name"],
                "side": u["side"],
                "series": u["series"],
                "period": u["period"],
                "title": u["title"],
                "url": u["url"],
                "date": u.get("date", ""),
            })

    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(out_rows)

    with open(REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        w.writeheader()
        w.writerows(review_rows)

    logger.info(f"완료: {OUT_CSV.name}에 {len(out_rows)}건 추가, {REVIEW_CSV.name}에 검토대상 {len(review_rows)}건")


if __name__ == "__main__":
    main()
