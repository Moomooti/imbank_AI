"""Phase 4-A 2단계 — 언어별 마스킹 정책표 (confidence-gated P2 확정).

Phase 1(COMET-Kiwi)~2(LaBSE)에서 확인된 구조적 결과를 정책으로 확정한다:
    - 베트남/인니/태국(DeepL 백엔드): 마스킹이 평균적으로 순이득 -> 기존대로 전체 마스킹
    - 필리핀/미얀마(NLLB 백엔드): 마스킹이 평균적으로 순손해 -> 기본은 비마스킹으로
      전환하고, 이번에 "복구 + baseline 능가"까지 확인된 용어만 예외적으로 마스킹
      허용(confidence-gated). "오염 후보로 안 걸렸던" 나머지 용어들도 검증된 적이
      없으므로 기본값은 비마스킹 -- 확인 안 된 이득을 가정하지 않는다.

산출물: masking_policy.csv (언어별 정책 요약) + mask_allowlist_{lang}.csv
(confidence-gated 언어의 실제 허용 용어 목록, app/terms에서 참조 가능한 형태)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "glossary" / "financial_terms_v2.csv"

FULL_MASK_LANGS = ["vie_Latn", "ind_Latn", "tha_Thai"]
GATED_LANGS = {"tgl_Latn": "tgl", "mya_Mymr": "mya"}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(V2_CSV, encoding="utf-8-sig")

    policy_rows = []
    for lang in FULL_MASK_LANGS:
        policy_rows.append(
            {
                "lang": lang,
                "policy": "full_mask",
                "n_terms_maskable": len(df),
                "n_terms_total": len(df),
                "note": "Phase 1~2에서 마스킹이 평균 순이득으로 확인됨(COMET-Kiwi diff, LaBSE diff 둘 다 양수) -- 기존 정책 유지",
            }
        )

    for lang, prefix in GATED_LANGS.items():
        status_col = f"{prefix}_repair_status"
        allowlist = df[df[status_col].notna()][["ko_term", "match_priority", f"{prefix}_draft", status_col]].copy()
        allowlist_path = ROOT / "data" / "masking" / f"mask_allowlist_{lang}.csv"
        allowlist.to_csv(allowlist_path, index=False, encoding="utf-8-sig")

        policy_rows.append(
            {
                "lang": lang,
                "policy": "confidence_gated (default=no_mask)",
                "n_terms_maskable": len(allowlist),
                "n_terms_total": len(df),
                "note": (
                    "Phase 1~2에서 마스킹이 평균 순손해로 확인됨 -- 기본은 비마스킹(순정 NLLB 그대로). "
                    f"이번에 복구+baseline 능가까지 검증된 {len(allowlist)}개 용어만 예외적으로 마스킹 허용 "
                    f"(mask_allowlist_{lang}.csv). 나머지 용어는 검증 이력이 없으므로 비마스킹."
                ),
            }
        )

    policy_df = pd.DataFrame(policy_rows)
    policy_path = ROOT / "data" / "masking" / "masking_policy.csv"
    policy_df.to_csv(policy_path, index=False, encoding="utf-8-sig")

    print("=== 언어별 마스킹 정책표 ===")
    print(policy_df.to_string(index=False))
    print(f"\n저장: {policy_path.name}")
    for lang in GATED_LANGS:
        print(f"저장: data/masking/mask_allowlist_{lang}.csv")


if __name__ == "__main__":
    main()
