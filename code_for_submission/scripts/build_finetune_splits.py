"""Phase 5 마무리 — 파인튜닝 데이터 최종 구성 (금융 도메인 어댑터용).

분리형 아키텍처 반영:
    - 이 대조 데이터(contrastive_pairs.csv) = [금융 도메인 어댑터] 학습용.
      "동음이의어를 금융/비금융으로 구별한다"는 언어 무관 핵심 IP.
    - [언어 어댑터](언어별 유창성)는 별도 데이터로 나중에 준비 -- 여기서 안 다룸.

분리 기준: **문장 단위가 아니라 term_id(용어) 단위**로 train/val/test를 나눈다.
같은 용어의 다른 문장이 train과 test에 걸쳐 있으면, 모델이 "이 용어는 이렇게
번역한다"를 통째로 암기해버려도 test에서 그대로 맞아버려 과대평가된다 --
진짜 보고 싶은 건 "새로운 문맥에서도 이 용어의 감을 잡는지"이기 때문에
용어 자체를 통째로 분리해야 한다.

1등급(한자 동음이의어)/2등급(빈도 편향형) 비율을 유지하며 약 70/15/15로
용어를 나눈다(용어 수가 20개뿐이라 무작위 대신 고정 시드로 재현 가능하게).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np              
import pandas as pd              

ROOT = Path(__file__).resolve().parents[1]
PAIRS_CSV = ROOT / "data" / "contrastive" / "contrastive_pairs.csv"
TARGETS_CSV = ROOT / "data" / "contrastive" / "contrastive_targets.csv"

SEED = 42


def split_terms_by_grade(targets: pd.DataFrame) -> dict[str, str]:
    """등급별로 계층화해서 용어를 train/val/test에 배정 (약 70/15/15)."""
    rng = np.random.default_rng(SEED)
    assignment: dict[str, str] = {}
    for grade, group in targets.groupby("grade"):
        terms = group["ko_term"].tolist()
        rng.shuffle(terms)
        n = len(terms)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        for t in terms[:n_train]:
            assignment[t] = "train"
        for t in terms[n_train : n_train + n_val]:
            assignment[t] = "val"
        for t in terms[n_train + n_val :]:
            assignment[t] = "test"
    return assignment


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
    targets = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig")

    usable = df[df["translation"].notna() & df["labse_filter_pass"]].copy()
    print(f"사용 가능 행: {len(usable)}/{len(df)}")

    assignment = split_terms_by_grade(targets)
    usable["split"] = usable["term"].map(assignment)

    missing = usable[usable["split"].isna()]
    if len(missing):
        print(f"경고: split 배정 안 된 용어 {missing['term'].unique().tolist()} -- train으로 폴백")
        usable["split"] = usable["split"].fillna("train")

    print("\n=== 용어 배정 (등급별) ===")
    assign_df = pd.DataFrame(
        [{"ko_term": t, "split": s} for t, s in assignment.items()]
    ).merge(targets[["ko_term", "grade"]], on="ko_term")
    print(assign_df.sort_values(["split", "grade"]).to_string(index=False))

    print("\n=== split별 용어 수 / 행 수 ===")
    print(assign_df.groupby(["split", "grade"]).size().unstack(fill_value=0).to_string())
    print(usable.groupby("split").size().to_string())

    for split in ["train", "val", "test"]:
        sub = usable[usable["split"] == split].drop(columns=["split"])
        out_path = ROOT / "data" / "contrastive" / f"contrastive_{split}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"저장: {out_path.name} ({len(sub)}행, {sub['term'].nunique()}개 용어)")

    assign_df.to_csv(ROOT / "data" / "contrastive" / "contrastive_split_terms.csv", index=False, encoding="utf-8-sig")
    print("저장: contrastive_split_terms.csv (용어->split 매핑)")


if __name__ == "__main__":
    main()
