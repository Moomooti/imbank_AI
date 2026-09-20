"""Phase 3 — 용어 치환용 사전 인덱스.

financial_terms_translated.csv(언어별 초벌 번역, Phase 1 작업 A)와
financial_terms_categorized.csv(category/related_category/cat_confidence, 최근 갱신)를
match_priority + ko_term 기준으로 합쳐서, 언어별 "치환기가 찾아야 할 표제어 문자열"
목록을 만든다.

D그룹(필리핀)은 행마다 이미 tgl_src로 loanword 여부가 기록돼 있다(Phase 1 작업 A에서
is_loanword=True인 행은 tgl_draft=eng, tgl_src="glossary"로 저장됨). 그래서 별도
재판정 없이 tgl_src=="glossary" 여부만 보면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATEGORIZED_CSV = ROOT / "data" / "glossary" / "financial_terms_categorized.csv"
TRANSLATED_CSV_V2 = ROOT / "data" / "glossary" / "financial_terms_v2.csv"
TRANSLATED_CSV_V1 = ROOT / "data" / "glossary" / "financial_terms_translated.csv"
# v2(Phase 4-A 복구 seed 반영본)가 있으면 그걸 쓰고, 없으면(구버전 체크아웃 등)
# v1으로 폴백한다.
TRANSLATED_CSV = TRANSLATED_CSV_V2 if TRANSLATED_CSV_V2.exists() else TRANSLATED_CSV_V1

# FLORES 코드 -> {lang}_draft/{lang}_src 컬럼 접두사
LANG_PREFIX = {
    "vie_Latn": "vie",
    "ind_Latn": "ind",
    "tha_Thai": "tha",
    "mya_Mymr": "mya",
    "tgl_Latn": "tgl",
}

LATIN_LANGS = {"vie_Latn", "ind_Latn", "tgl_Latn"}  # 정규식 경계 매칭 (분절기 불필요)
TOKENIZED_LANGS = {"tha_Thai", "mya_Mymr"}  # Phase 2 분절기 필요

MVP_CATEGORIES = ["계좌", "이체", "해외송금·외환", "예금상품", "카드·전자금융", "금융 지원"]


@dataclass(frozen=True)
class GlossaryRow:
    ko_term: str
    match_priority: int
    term_text: str  # 이 언어에서 "치환기가 찾아야 할" 문자열 (없으면 빈 문자열)
    is_loanword: bool  # D그룹에서만 의미 있음; 그 외 언어는 항상 False


def load_merged(mvp_only: bool = False) -> pd.DataFrame:
    cat = pd.read_csv(CATEGORIZED_CSV, encoding="utf-8-sig")
    tr = pd.read_csv(TRANSLATED_CSV, encoding="utf-8-sig")
    merged = tr.merge(
        cat[["match_priority", "ko_term", "category", "related_category", "cat_confidence"]],
        on=["match_priority", "ko_term"],
        how="left",
        validate="one_to_one",
    )
    if mvp_only:
        merged = merged[merged["category"].isin(MVP_CATEGORIES)].reset_index(drop=True)
    return merged


def build_rows(df: pd.DataFrame, lang: str) -> list[GlossaryRow]:
    prefix = LANG_PREFIX[lang]
    draft_col, src_col = f"{prefix}_draft", f"{prefix}_src"
    rows: list[GlossaryRow] = []
    for r in df.itertuples(index=False):
        draft = getattr(r, draft_col)
        if pd.isna(draft) or not str(draft).strip():
            continue
        is_loanword = lang == "tgl_Latn" and getattr(r, src_col) == "glossary"
        rows.append(
            GlossaryRow(
                ko_term=r.ko_term,
                match_priority=r.match_priority,
                term_text=str(draft),
                is_loanword=is_loanword,
            )
        )
    return rows
