"""Phase 4-1 마무리 — 유실/오염 필터링해서 깨끗한 학습 페어만 추출.

training_pairs_{lang}.csv 5개에서 아래 3개 조건을 전부 만족하는 행만 남긴다
(Phase 4 미얀마 파인튜닝 등의 학습 입력이 됨):

    1) terms_used가 비어있지 않음
       -> 사전에 없는 용어라 애초에 마스킹이 안 걸린 문장 제외 (51건, 언어 무관)
    2) identifier_survival_rate == 1.0
       -> 식별자가 하나라도 유실된 문장은 통째로 배제(부분 유실 포함)
    3) max_suspicion_score < 50
       -> 역번역 대조로 고위험 오역 의심되는 용어가 섞인 문장 제외

깔때기(funnel) 방식으로 순서대로 배제해서 각 단계 배제 건수를 리포트한다
(한 행이 여러 조건에 동시 걸릴 수 있어 사유별 합산이 아니라 순차 배제로 계산).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd              

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["vie_Latn", "ind_Latn", "tha_Thai", "tgl_Latn", "mya_Mymr"]
SUSPICION_THRESHOLD = 50


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    summary_rows = []
    for lang in LANGS:
        src = ROOT / "data" / "training_pairs" / f"training_pairs_{lang}.csv"
        df = pd.read_csv(src, encoding="utf-8-sig")
        total = len(df)

        has_terms = df["terms_used"].notna() & (df["terms_used"].astype(str).str.strip() != "")
        no_terms_excluded = total - has_terms.sum()
        step1 = df[has_terms]

        survived = step1["identifier_survival_rate"] == 1.0
        lost_excluded = len(step1) - survived.sum()
        step2 = step1[survived]

        clean = step2["max_suspicion_score"] < SUSPICION_THRESHOLD
        suspicious_excluded = len(step2) - clean.sum()
        final = step2[clean]

        out_path = ROOT / "data" / "training_pairs" / f"training_pairs_clean_{lang}.csv"
        final.to_csv(out_path, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "lang": lang,
                "total": total,
                "excl_no_terms": no_terms_excluded,
                "excl_identifier_lost": lost_excluded,
                "excl_suspicious(>=50)": suspicious_excluded,
                "clean_final": len(final),
                "clean_rate": f"{len(final) / total:.1%}",
            }
        )
        print(f"저장: {out_path.name} ({len(final)}행)")

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("언어별 채택/제외 리포트 (순차 배제, funnel)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    summary_path = ROOT / "data" / "training_pairs" / "training_pairs_clean_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n요약 저장: {summary_path.name}")


if __name__ == "__main__":
    main()
