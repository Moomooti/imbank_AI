"""Phase 4-A 2단계 — glossary v2 확정 (baseline을 넘은 승인 seed만 반영).

Phase 4-A 파일럿/확장에서 "approved"(COMET+LaBSE 둘 다 오염된 파이프라인보다
상승)된 68건이라도, 그 중 상당수는 순정 NLLB baseline에는 여전히 못 미쳤다
(필리핀 32건 중 12건만, 미얀마 36건 중 15건만 baseline 이상). "복구됐다"와
"순정보다 낫다"는 다른 질문이라, glossary v2에는 후자 기준만 반영한다 --
순정 NLLB보다 나쁜 seed를 넣으면 마스킹 자체가 손해이기 때문이다.

반영 기준: verdict == "approved" AND repaired_comet_score >= nllb_baseline_score
    - repaired_comet_score > nllb_baseline_score (오차범위 밖으로 확실히 능가)
      -> status = "repaired_and_beaten"
    - repaired_comet_score ~= nllb_baseline_score (오차범위 내 동률)
      -> status = "baseline_only" ("동률 이상이면 반영"의 "동률" 쪽)
    - 그 외(순정보다 못함) -> 반영 안 함, 기존 seed(오염 상태) 그대로 유지
    - abstain 31건은 애초에 검토 대상 아님 (approved만 후보)

산출물: financial_terms_v2.csv, 언어별 반영/제외 건수 리포트.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd              

ROOT = Path(__file__).resolve().parents[1]

TRANSLATED_CSV = ROOT / "data" / "glossary" / "financial_terms_translated.csv"
REPAIR_CSV = ROOT / "data" / "glossary" / "repair_seed_166_checkpoint.csv"
CONFIRMED_CSV = ROOT / "data" / "glossary" / "confirmed_contamination_candidates.csv"
OUT_CSV = ROOT / "data" / "glossary" / "financial_terms_v2.csv"

LANG_PREFIX = {"tgl_Latn": "tgl", "mya_Mymr": "mya"}
TIE_EPSILON = 0.005                


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    repair = pd.read_csv(REPAIR_CSV, encoding="utf-8-sig")
    confirmed = pd.read_csv(CONFIRMED_CSV, encoding="utf-8-sig")
    merged = repair.merge(
        confirmed[["sentence_id", "lang", "nllb_baseline_score"]], on=["sentence_id", "lang"], how="left"
    )

    approved = merged[merged["verdict"] == "approved"].copy()
    diff = approved["repaired_comet_score"] - approved["nllb_baseline_score"]
    approved["status"] = "excluded_below_baseline"
    approved.loc[diff > TIE_EPSILON, "status"] = "repaired_and_beaten"
    approved.loc[diff.abs() <= TIE_EPSILON, "status"] = "baseline_only"

    reflect = approved[approved["status"].isin(["repaired_and_beaten", "baseline_only"])].copy()

    print("=== 반영 기준 적용 결과 ===")
    for lang in ["tgl_Latn", "mya_Mymr"]:
        n_total = len(approved[approved["lang"] == lang])
        n_reflect = len(reflect[reflect["lang"] == lang])
        n_beaten = len(reflect[(reflect["lang"] == lang) & (reflect["status"] == "repaired_and_beaten")])
        n_tie = len(reflect[(reflect["lang"] == lang) & (reflect["status"] == "baseline_only")])
        n_excluded = n_total - n_reflect
        print(f"{lang}: approved {n_total}건 중 반영 {n_reflect}건 "
              f"(능가 {n_beaten} + 동률 {n_tie}), baseline 미달로 제외 {n_excluded}건")

                                                                                        
    reflect = reflect.sort_values("repaired_comet_score", ascending=False).drop_duplicates(
        subset=["target_term", "lang"], keep="first"
    )
    print(f"\n(ko_term, lang) 중복 제거 후 실제 반영 항목 수: {len(reflect)}건")

                              
    df = pd.read_csv(TRANSLATED_CSV, encoding="utf-8-sig")
    for prefix in LANG_PREFIX.values():
        for col in [f"{prefix}_repair_status", f"{prefix}_repair_source",
                    f"{prefix}_comet_before", f"{prefix}_comet_after",
                    f"{prefix}_labse_before", f"{prefix}_labse_after"]:
            if col not in df.columns:
                df[col] = pd.NA

    applied_log = []
    not_found = []
    for r in reflect.itertuples(index=False):
        prefix = LANG_PREFIX[r.lang]
        idx = df.index[df["ko_term"] == r.target_term]
        if len(idx) == 0:
            not_found.append((r.target_term, r.lang))
            continue
        idx = idx[0]
        df.at[idx, f"{prefix}_draft"] = r.candidate
        df.at[idx, f"{prefix}_src"] = f"llm_repair_{r.provider}"
        df.at[idx, f"{prefix}_repair_status"] = r.status
        df.at[idx, f"{prefix}_repair_source"] = r.provider
        df.at[idx, f"{prefix}_comet_before"] = r.pipeline_score
        df.at[idx, f"{prefix}_comet_after"] = r.repaired_comet_score
        df.at[idx, f"{prefix}_labse_before"] = r.pipeline_labse_sim
        df.at[idx, f"{prefix}_labse_after"] = r.repaired_labse_sim
        applied_log.append((r.target_term, r.lang, r.status))

    if not_found:
        print(f"\n경고: glossary에서 못 찾은 target_term {len(not_found)}건 (반영 스킵): {not_found}")

    tmp = OUT_CSV.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(OUT_CSV)
    print(f"\n저장: {OUT_CSV.name} ({len(df)}행, 실반영 {len(applied_log)}건)")

    print("\n=== 최종 반영 로그 ===")
    for term, lang, status in applied_log:
        print(f"  {lang:10s} {term:20s} {status}")


if __name__ == "__main__":
    main()
