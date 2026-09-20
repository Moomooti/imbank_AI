"""Phase 5 후속 — 대조 데이터 번역문에서 "괄호+영어 병기" 습관 제거.

LLM이 낯선 금융 전문용어를 번역할 때 확신이 없으면 "hạn mức... (clearing)"처럼
영어 원어를 괄호로 같이 써버리는 경향이 있었다(504건 중 34건, 6.7%). 이대로
파인튜닝하면 모델이 "번역할 때마다 영어를 병기하는 습관"을 배울 위험이 있어
제거한다.

미얀마/태국은 비라틴 문자라 괄호 안에 라틴 문자가 있으면 무조건 외래어 삽입이
확실하므로 전부 제거. 베트남/인니/필리핀은 라틴 문자라 괄호 안 내용이 그
언어 자체의 정상적인 부연설명(예: 인니어 "sertifikat saham")일 수도 있어서,
실제로 관측된 영어 금융 전문용어 목록에 매칭되는 것만 제거한다(오탐 방지).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS_CSV = ROOT / "data" / "contrastive" / "contrastive_pairs.csv"

NON_LATIN_LANGS = {"mya_Mymr", "tha_Thai"}

# 실제 관측된 영어 금융 전문용어 병기 목록 (소문자 기준 매칭).
KNOWN_ENGLISH_GLOSSES = {
    "clearing", "call loan", "odd lot", "short selling", "call money",
    "inventory", "call option", "put option", "quote", "otc", "liquidity",
    "big bang", "private placement", "short sale", "derivatives", "settlement",
    "public offering", "shme", "position", "option",
}

PAREN_RE = re.compile(r"\s*\(([A-Za-z][A-Za-z\s\-']{1,40})\)")


def clean_translation(text: str, lang: str) -> tuple[str, int]:
    removed = 0

    def _sub(m: re.Match) -> str:
        nonlocal removed
        content = m.group(1).strip().lower()
        if lang in NON_LATIN_LANGS or content in KNOWN_ENGLISH_GLOSSES:
            removed += 1
            return ""
        return m.group(0)

    cleaned = PAREN_RE.sub(_sub, text)
    cleaned = re.sub(r"\s+([.,!?？。！，])", r"\1", cleaned)  # 제거로 생긴 구두점 앞 공백 정리
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, removed


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
    total_removed = 0
    changed_rows = 0
    for idx, row in df.iterrows():
        if pd.isna(row["translation"]):
            continue
        cleaned, n = clean_translation(str(row["translation"]), row["lang"])
        if n > 0:
            df.at[idx, "translation"] = cleaned
            total_removed += n
            changed_rows += 1

    tmp = PAIRS_CSV.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(PAIRS_CSV)

    print(f"정리된 병기 개수: {total_removed}건 (영향받은 행: {changed_rows}건)")
    print(f"저장 완료: {PAIRS_CSV.name}")


if __name__ == "__main__":
    main()
