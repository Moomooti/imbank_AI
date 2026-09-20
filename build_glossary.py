

import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "glossary" / "financial_terms_clean.csv"

FLAG_COLUMNS = ["is_short", "is_multiword", "is_dup"]


def _norm(s) -> str:
    """공백 차이를 무시하고 비교하기 위한 정규화."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", "", str(s))


def load_glossary(csv_path: Path = CSV_PATH) -> dict[str, dict]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in FLAG_COLUMNS:
        df[col] = df[col].fillna("") == "Y"

    df = df.sort_values("match_priority")

    glossary: dict[str, dict] = {}
    skipped_exact_dups = 0
    multi_sense_terms: list[str] = []

    for row in df.itertuples(index=False):
        sense = {
            "match_priority": row.match_priority,
            "hanja": row.hanja if pd.notna(row.hanja) else "",
            "eng": row.eng if pd.notna(row.eng) else "",
            "ko_def": row.ko_def if pd.notna(row.ko_def) else "",
            "raw_term": row.raw_term if pd.notna(row.raw_term) else "",
        }

        entry = glossary.get(row.ko_term)
        if entry is None:
            glossary[row.ko_term] = {
                "is_short": row.is_short,
                "is_multiword": row.is_multiword,
                "is_dup": row.is_dup,
                "senses": [sense],
            }
            continue

        existing_defs = {_norm(s["ko_def"]) for s in entry["senses"]}
        if _norm(sense["ko_def"]) in existing_defs:
            # 정의까지 동일한 완전중복 -> 흡수(skip)
            skipped_exact_dups += 1
            continue

        # 정의가 다른 다의어 -> senses에 추가 보존
        entry["senses"].append(sense)
        if row.ko_term not in multi_sense_terms:
            multi_sense_terms.append(row.ko_term)

    print(f"[glossary] 완전중복(정의 동일) {skipped_exact_dups}건 흡수")
    print(f"[glossary] 다의어(정의 상이) {len(multi_sense_terms)}건 보존: {multi_sense_terms}")

    return glossary


def primary_translation(entry: dict) -> dict:
    """단일 값이 필요할 때 대표 뜻(senses[0])을 반환."""
    return entry["senses"][0]


if __name__ == "__main__":
    glossary = load_glossary()
    print(f"고유 표제어 수: {len(glossary)}")

    multi = {k: v for k, v in glossary.items() if len(v["senses"]) > 1}
    for term, entry in multi.items():
        print(f"\n[{term}] senses={len(entry['senses'])}")
        for i, s in enumerate(entry["senses"], 1):
            print(f"  {i}. eng={s['eng']!r} ko_def={s['ko_def'][:40]!r}...")

    sample_term = next(iter(glossary))
    print(f"\n예시(단일 뜻): {sample_term} -> {primary_translation(glossary[sample_term])}")
