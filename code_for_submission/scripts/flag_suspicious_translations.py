"""Phase 3.5 확장 — "의심스러운 번역" 플래그 목록 생성 (핵심 산출물).

목적: 1298개 전체를 번역 "완성"하려는 게 아니라, 문맥 없이 단독으로 번역된
용어 중 오염됐을 가능성이 높은 것들을 대량으로 골라내 사람 검수 우선순위를
매기는 것. ("대변이체시스템" -> "배설물 이송 시스템" 사고가 우연이 아니라
동음이의어/다의어 용어에서 구조적으로 반복될 것이라는 가설을 검증)

의심 점수 휴리스틱 (가중치 합산, 0~100):
    1) is_short (2글자 이하, 49개)   : +30  — 동음이의어 오역 최고 위험군
    2) is_dup (다의어, 20개)          : +20  — 뜻이 여러 개라 오역 가능
    3) 역번역 대조                    : +40/+20 — {lang}_draft를 한국어로
       역번역해서 원래 ko_term과 얼마나 다른지(문자열 유사도). 유사도가
       낮을수록 "번역기가 완전히 딴 뜻으로 이해했다"는 신호.
       ("대변" -> "배설물" 사고가 바로 이 신호로 잡힘)
    4) 길이비 이상                    : +10  — 역번역 결과가 원문보다 극단적으로
       길거나 짧으면 엉뚱한 걸 번역했을 가능성 (진짜 사전 대조는 못 하므로 근사치)

역번역은 항상 NLLB로만 한다(DeepL 쿼터 절약 — 역번역은 정답이 아니라 의심
점수 참고용이라 품질 요구가 낮음). 필리핀어 loanword 통과 행(tgl_src=="glossary")은
애초에 "번역"이 아니라 영어를 그대로 통과시킨 것이므로 역번역 대조 대상에서
제외한다(제외해도 is_short/is_dup 신호는 그대로 반영됨).

중간저장/이어하기: 이미 역번역이 끝난 (ko_term, lang) 쌍은 건너뛴다.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mt.translate import back_translate_to_korean              

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flag_suspicious")

ROOT = Path(__file__).resolve().parents[1]
TRANSLATED_CSV = ROOT / "data" / "glossary" / "financial_terms_translated.csv"
WORK_CSV = ROOT / "data" / "qa_review" / "suspicious_work.csv"                      
OUT_CSV = ROOT / "data" / "qa_review" / "suspicious_translations.csv"

LANG_PREFIX = {"vie_Latn": "vie", "ind_Latn": "ind", "tha_Thai": "tha", "mya_Mymr": "mya", "tgl_Latn": "tgl"}
NLLB_BATCH_SIZE = 16

WORK_FIELDNAMES = ["ko_term", "match_priority", "lang", "term_text", "back_translation_ko"]


def load_work() -> pd.DataFrame:
    if WORK_CSV.exists():
        return pd.read_csv(WORK_CSV, encoding="utf-8-sig")
    return pd.DataFrame(columns=WORK_FIELDNAMES)


def save_work_row(rows: list[dict]) -> None:
    write_header = not WORK_CSV.exists()
    with open(WORK_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=WORK_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def back_translate_all(df: pd.DataFrame, limit: int | None) -> None:
    """모든 (ko_term, lang) 쌍에 대해 역번역을 채워서 WORK_CSV에 누적한다."""
    existing = load_work()
    done_keys = set(zip(existing["ko_term"], existing["lang"])) if len(existing) else set()

    for lang, prefix in LANG_PREFIX.items():
        draft_col, src_col = f"{prefix}_draft", f"{prefix}_src"
        candidates = df[df[draft_col].notna() & (df[draft_col].astype(str).str.strip() != "")]
        if lang == "tgl_Latn":
            candidates = candidates[candidates[src_col] != "glossary"]                          

        todo = [
            (r.ko_term, r.match_priority, str(getattr(r, draft_col)))
            for r in candidates.itertuples(index=False)
            if (r.ko_term, lang) not in done_keys
        ]
        if limit:
            todo = todo[:limit]
        if not todo:
            logger.info(f"[{lang}] 역번역 대상 없음 (이미 완료됨)")
            continue
        logger.info(f"[{lang}] 역번역 대상 {len(todo)}건")

        for i, chunk in enumerate(chunked(todo, NLLB_BATCH_SIZE)):
            texts = [c[2] for c in chunk]
            try:
                back = back_translate_to_korean(texts, source_lang=lang)
            except Exception as e:                
                logger.error(f"[{lang}] 역번역 배치 실패, 스킵: {e!r}")
                continue
            rows = [
                {"ko_term": ko_term, "match_priority": mp, "lang": lang, "term_text": term_text, "back_translation_ko": bt}
                for (ko_term, mp, term_text), bt in zip(chunk, back)
            ]
            save_work_row(rows)
            done = min((i + 1) * NLLB_BATCH_SIZE, len(todo))
            logger.info(f"[{lang}] {done}/{len(todo)} 역번역 완료 (누적 저장됨)")


def score_row(ko_term: str, back_ko: str, is_short: bool, is_dup: bool) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if is_short:
        score += 30
        reasons.append("is_short(2글자 이하)")
    if is_dup:
        score += 20
        reasons.append("is_dup(다의어)")

    similarity = SequenceMatcher(None, ko_term, back_ko).ratio()
    if similarity < 0.3:
        score += 40
        reasons.append(f"역번역 유사도 낮음({similarity:.2f})")
    elif similarity < 0.5:
        score += 20
        reasons.append(f"역번역 유사도 다소낮음({similarity:.2f})")

    len_ratio = len(back_ko) / max(len(ko_term), 1)
    if len_ratio > 3 or len_ratio < 0.34:
        score += 10
        reasons.append(f"길이비 이상({len_ratio:.2f})")

    return score, reasons, similarity, len_ratio


def build_report() -> pd.DataFrame:
    df = pd.read_csv(TRANSLATED_CSV, encoding="utf-8-sig")
                                                                    
                                                                  
    flags = df.drop_duplicates(subset="ko_term", keep="first").set_index("ko_term")[["is_short", "is_dup"]]
    flags["is_short"] = flags["is_short"].fillna("") == "Y"
    flags["is_dup"] = flags["is_dup"].fillna("") == "Y"

    work = load_work()
    out_rows = []
    for r in work.itertuples(index=False):
        is_short = bool(flags.loc[r.ko_term, "is_short"]) if r.ko_term in flags.index else False
        is_dup = bool(flags.loc[r.ko_term, "is_dup"]) if r.ko_term in flags.index else False
        score, reasons, similarity, len_ratio = score_row(r.ko_term, str(r.back_translation_ko), is_short, is_dup)
        out_rows.append({
            "ko_term": r.ko_term,
            "lang": r.lang,
            "term_text": r.term_text,
            "back_translation_ko": r.back_translation_ko,
            "similarity": round(similarity, 3),
            "len_ratio": round(len_ratio, 2),
            "is_short": is_short,
            "is_dup": is_dup,
            "suspicion_score": score,
            "reasons": "; ".join(reasons) if reasons else "",
        })

    out_df = pd.DataFrame(out_rows).sort_values(["suspicion_score", "ko_term"], ascending=[False, True])
    return out_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="언어당 역번역 대상 수 제한 (테스트용)")
    parser.add_argument("--skip-back-translate", action="store_true", help="역번역 없이 기존 WORK_CSV로만 리포트 재생성")
    args = parser.parse_args()

    df = pd.read_csv(TRANSLATED_CSV, encoding="utf-8-sig")
    logger.info(f"대상 용어 수: {len(df)}")

    if not args.skip_back_translate:
        back_translate_all(df, args.limit)

    report = build_report()
    report.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"완료: {OUT_CSV.name} 저장됨 ({len(report)}행)")

    logger.info(f"\n상위 20개 의심 사례:\n" + report.head(20).to_string())
    logger.info(f"\n언어별 평균 의심점수:\n{report.groupby('lang')['suspicion_score'].mean().to_string()}")
    logger.info(f"\nsuspicion_score > 0 인 행: {(report['suspicion_score'] > 0).sum()} / {len(report)}")


if __name__ == "__main__":
    main()
