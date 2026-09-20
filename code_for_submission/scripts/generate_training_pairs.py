"""Phase 4-1 — 학습 소스 문장 준비 (mask -> translate -> restore 재사용).

팀원이 만든 1,482개 한국어 문장 템플릿(검수 완료, "최종사용문장" 컬럼)을
Phase 3.5 마스킹 파이프라인에 통과시켜 (한국어 소스, 용어가 정확히 박힌
목표어 문장) 페어를 5개 언어로 만든다.

주의 (사용자 지시 그대로):
- 이 소스 문장셋은 "학습 소스"로만 쓴다. gold 평가셋 아님 (기존 합의).
- 용어 번역 자체는 여전히 seed 품질(대변->배설물 같은 오염 가능) — 그래서
  suspicious_translations.csv의 의심점수를 그대로 실어서, 나중에 학습
  데이터 필터링에 쓸 수 있게 한다.
- 미얀마는 Phase 3에서 확인된 "...ပြီး+용어" 연결 시 글자 누락 잔여위험이
  있어(커스텀딕셔너리로 못 고침), 식별자 생존율을 문장 단위로 별도 컬럼에
  기록하고 유실 문장을 따로 뽑아 보여준다.

번역 백엔드는 Task A와 동일한 프로덕션 조합을 그대로 씀(왕복검증 때처럼
한 언어를 두 백엔드로 중복 테스트하지 않음): vi/id/th=DeepL, tgl/mya=NLLB.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd              
from dotenv import load_dotenv              

from app.mt.translate import translate_batch              
from app.terms import build_rows, load_merged, mask_text, restore_text              

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

                                                     
                                                 
                          
SOURCE_DIR = Path(r"C:\Users\yues7\Downloads")
SOURCE_NFC_NAME = "한국어_금융문장_후보셋_최종.csv"
LOCAL_SOURCE_COPY = ROOT / "data" / "source" / SOURCE_NFC_NAME

SUSPICIOUS_CSV = ROOT / "data" / "qa_review" / "suspicious_translations.csv"

                                                  
LANG_TARGETS = {
    "vie_Latn": ("deepl", "VI"),
    "ind_Latn": ("deepl", "ID"),
    "tha_Thai": ("deepl", "TH"),
    "tgl_Latn": ("nllb", None),
    "mya_Mymr": ("nllb", None),
}

BATCH_SIZE = 16


def resolve_source_path() -> Path:
    if LOCAL_SOURCE_COPY.exists():
        return LOCAL_SOURCE_COPY
    for name in os.listdir(SOURCE_DIR):
        if name.lower().endswith(".csv") and unicodedata.normalize("NFC", name) == SOURCE_NFC_NAME:
            shutil.copy(SOURCE_DIR / name, LOCAL_SOURCE_COPY)
            return LOCAL_SOURCE_COPY
    raise FileNotFoundError(f"{SOURCE_NFC_NAME}를 {SOURCE_DIR}에서 찾을 수 없습니다.")


def load_suspicious_lookup() -> dict[tuple[str, str], float]:
    """(ko_term, lang) -> suspicion_score. 리포트 없으면 빈 dict(전부 0점 취급)."""
    if not SUSPICIOUS_CSV.exists():
        return {}
    df = pd.read_csv(SUSPICIOUS_CSV, encoding="utf-8-sig")
    return {(r.ko_term, r.lang): r.suspicion_score for r in df.itertuples(index=False)}


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
                                                     
                                                           
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="소스 문장 수 제한 (스모크 테스트용)")
    parser.add_argument("--langs", nargs="*", default=list(LANG_TARGETS.keys()), choices=list(LANG_TARGETS.keys()))
    parser.add_argument("--suffix", default="", help="출력 파일명에 붙일 접미사 (예: _smoke50)")
    parser.add_argument(
        "--known-terms-file", default=None, help="한 줄에 하나씩 금융용어를 적은 파일 -- 이 문장들만 처리 (부분 재검증용)"
    )
    args = parser.parse_args()

    src_path = resolve_source_path()
    src_df = pd.read_csv(src_path, encoding="utf-8-sig")
    if args.known_terms_file:
        wanted = {
            line.strip()
            for line in Path(args.known_terms_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        src_df = src_df[src_df["금융용어"].isin(wanted)]
    if args.limit:
        src_df = src_df.head(args.limit)
    print(f"소스 문장 수: {len(src_df)} (원본 {resolve_source_path().name})")

    glossary_df = load_merged(mvp_only=False)
    ko_terms_all = list(zip(glossary_df["ko_term"], glossary_df["match_priority"]))
    rows_by_lang = {lang: build_rows(glossary_df, lang) for lang in LANG_TARGETS}
    suspicious = load_suspicious_lookup()
    print(f"의심 리포트 로드: {len(suspicious)}쌍 (없으면 전부 0점 처리)")

    src_rows = list(src_df.itertuples(index=False))

    for lang, (backend, deepl_lang) in LANG_TARGETS.items():
        if lang not in args.langs:
            continue
        print(f"\n=== {lang} ({backend}) ===")

                                                             
                                                        
                                                  
        masked_records = []
        for row in src_rows:
            ko_source = str(row.최종사용문장)
            masked, entries = mask_text(ko_source, ko_terms_all, lang=lang)
            masked_records.append(
                {
                    "sentence_id": row.문장ID,
                    "ko_source": ko_source,
                    "known_term": row.금융용어,
                    "category": row.업무카테고리,
                    "related_category": getattr(row, "관련카테고리", None),
                    "masked": masked,
                    "entries": entries,
                }
            )

        n_no_match = sum(1 for r in masked_records if not r["entries"])
        n_known_missing = sum(
            1 for r in masked_records if r["known_term"] not in [e.ko_term for e in r["entries"]]
        )
        print(f"  용어 전혀 미검출 문장: {n_no_match}/{len(masked_records)}")
        print(f"  금융용어 컬럼값이 마스킹 결과에 안 잡힌 문장: {n_known_missing}/{len(masked_records)}")

        out_rows = []
        done = 0
        for chunk in chunked(masked_records, BATCH_SIZE):
            texts = [r["masked"] for r in chunk]
            translated_list = translate_batch(texts, tgt_lang=lang, backend=backend, deepl_lang=deepl_lang)
            for r, translated in zip(chunk, translated_list):
                if r["entries"]:
                    restored_text, report = restore_text(translated, r["entries"], rows_by_lang[lang])
                else:
                    restored_text, report = translated, None

                terms_used = [e.ko_term for e in r["entries"]]
                scores = [suspicious.get((t, lang), 0) or 0 for t in terms_used]
                max_score = max(scores) if scores else 0
                flagged_terms = [t for t, s in zip(terms_used, scores) if s and s > 0]
                lost_terms = [d["ko_term"] for d in report.details if d["status"] == "lost"] if report else []

                out_rows.append(
                    {
                        "sentence_id": r["sentence_id"],
                        "ko_source": r["ko_source"],
                        "target_sentence": restored_text,
                        "terms_used": ";".join(terms_used),
                        "known_term": r["known_term"],
                        "known_term_matched": r["known_term"] in terms_used,
                        "category": r["category"],
                        "related_category": r["related_category"],
                        "identifier_survival_rate": report.survival_rate if report else 1.0,
                        "identifier_order_preserved_rate": report.order_preservation_rate if report else 1.0,
                        "restore_success_rate": report.restore_success_rate if report else 1.0,
                        "fallback_used": report.fallback_used if report else 0,
                        "lost_terms": ";".join(lost_terms),
                        "max_suspicion_score": max_score,
                        "flagged_terms": ";".join(flagged_terms),
                    }
                )
            done += len(chunk)
            if done % (BATCH_SIZE * 5) == 0 or done == len(masked_records):
                print(f"  {done}/{len(masked_records)} 번역 완료")

        out_df = pd.DataFrame(out_rows)
        out_path = ROOT / "data" / "training_pairs" / f"training_pairs_{lang}{args.suffix}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"저장: {out_path} ({len(out_df)}행)")
        print(f"  known_term_matched=False: {(~out_df['known_term_matched']).sum()}건")
        print(f"  identifier_survival_rate<1.0: {(out_df['identifier_survival_rate'] < 1.0).sum()}건")
        print(f"  max_suspicion_score>0: {(out_df['max_suspicion_score'] > 0).sum()}건")

        if lang == "mya_Mymr":
            low = out_df[out_df["identifier_survival_rate"] < 1.0]
            if len(low):
                print(f"\n[미얀마] 식별자 유실 문장 상세 ({len(low)}건, 최대 10건):")
                print(low[["sentence_id", "ko_source", "terms_used", "target_sentence"]].head(10).to_string())


if __name__ == "__main__":
    main()
