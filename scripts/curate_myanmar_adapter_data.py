"""Phase 7 — 미얀마 언어 어댑터 학습 데이터 큐레이션 (규칙 기반, LaBSE 제외).

opus_ko_my.csv(630,678쌍, OpenSubtitles/TED2020/bible-uedin)에서 금융
서비스 톤에 맞게 소스 비중을 조절하고 잔여 노이즈를 제거한다.

LaBSE 교차언어 정렬 검사는 CPU로 127K쌍에 2.7시간이 걸려 이번엔 생략 --
규칙 기반 필터만으로 빠르게 마무리하고, 정렬 정밀검사는 나중에 품질 문제가
실제로 보이면 GPU로 재검토한다.

소스 전략:
    - TED2020(격식·설명체): 전량 사용, 최우선 -- 금융 어시스턴트 톤에 가장 가까움
    - OpenSubtitles(구어·반말): 정제 후 목표치를 채우는 만큼만 무작위 샘플링
    - bible-uedin(고어체): 완전 제외 -- "~하시니라" 말투 방지

규칙 필터:
    1) 빈 값/중복 제거
    2) Zawgyi 잔여 최종 제거 (myanmartools)
    3) 길이비 이상 제거 (my_len/ko_len 3배 초과 등 극단치)
    4) 숫자/특수문자만 있는 줄 제거
    5) OpenSubtitles 한정: 최소 길이 미달(감탄사성) 제외 + 슬랭 블록리스트
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from myanmartools import ZawgyiDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "data" / "myanmar_adapter" / "opus_ko_my.csv"
OUT_CSV = ROOT / "data" / "myanmar_adapter" / "my_adapter_train.csv"
REPORT_CSV = ROOT / "data" / "myanmar_adapter" / "my_adapter_curation_report.csv"

RATIO_MAX = 3.0  # my_len/ko_len 이 이걸 넘거나 1/이걸 밑돌면 제거 (대칭 상한)
OPENSUB_MIN_KO_LEN = 10
ZAWGYI_THRESHOLD = 0.9
TARGET_TOTAL = 90000
RNG_SEED = 42

SLANG_BLOCKLIST = {"젠장", "젠장할", "빌어먹을", "닥쳐", "닥쳐!", "씨발", "개새끼", "죽어", "죽어!"}

# 숫자/기호/공백만으로 이루어진 줄 (문장이라 부를 게 없는 것)
ONLY_SYMBOLIC_RE = re.compile(r"^[\d\s\W]+$", re.UNICODE)


def is_only_symbolic(text: str) -> bool:
    # 한글/미얀마 등 문자(letter) 카테고리가 하나도 없으면 True
    return not any(c.isalpha() for c in text)


def apply_common_filters(df: pd.DataFrame, detector: ZawgyiDetector) -> pd.DataFrame:
    df = df.dropna(subset=["ko", "my"]).copy()
    df["ko"] = df["ko"].astype(str).str.strip()
    df["my"] = df["my"].astype(str).str.strip()
    df = df[(df["ko"] != "") & (df["my"] != "")]
    df = df.drop_duplicates(subset=["ko", "my"])

    df = df[~df["ko"].apply(is_only_symbolic)]
    df = df[~df["my"].apply(is_only_symbolic)]

    df["ko_len"] = df["ko"].str.len()
    df["my_len"] = df["my"].str.len()
    df["ratio"] = df["my_len"] / df["ko_len"].clip(lower=1)
    df = df[(df["ratio"] >= 1 / RATIO_MAX) & (df["ratio"] <= RATIO_MAX)]

    df["zawgyi_prob"] = df["my"].apply(lambda t: detector.get_zawgyi_probability(t))
    df = df[df["zawgyi_prob"] < ZAWGYI_THRESHOLD]

    return df


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(SRC_CSV, encoding="utf-8-sig")
    print(f"원본 총: {len(df)}쌍")
    by_source = {name: g.copy() for name, g in df.groupby("source")}
    detector = ZawgyiDetector()

    n_bible = len(by_source.get("bible-uedin", []))
    print(f"bible-uedin {n_bible}쌍 -- 제외(고어체)")

    ted = apply_common_filters(by_source["TED2020"], detector)
    n_ted_before = len(by_source["TED2020"])
    print(f"TED2020: {n_ted_before} -> 필터 후 {len(ted)}쌍 (전량 채택)")

    opensub = apply_common_filters(by_source["OpenSubtitles"], detector)
    n_opensub_before = len(by_source["OpenSubtitles"])
    n_after_common = len(opensub)
    opensub = opensub[opensub["ko_len"] >= OPENSUB_MIN_KO_LEN]
    opensub = opensub[~opensub["ko"].isin(SLANG_BLOCKLIST)]
    n_after_tone = len(opensub)
    print(f"OpenSubtitles: {n_opensub_before} -> 공통필터 {n_after_common} -> 톤/길이 필터 {n_after_tone}")

    need = max(TARGET_TOTAL - len(ted), 0)
    rng = np.random.default_rng(RNG_SEED)
    if len(opensub) > need:
        idx = rng.choice(len(opensub), size=need, replace=False)
        opensub_sampled = opensub.iloc[idx]
    else:
        opensub_sampled = opensub
    print(f"OpenSubtitles 최종 샘플링: {len(opensub_sampled)}쌍 (목표 총량 {TARGET_TOTAL} 채우기 위해 {need}개 필요)")

    ted_out = ted[["ko", "my"]].copy()
    ted_out["source"] = "TED2020"
    opensub_out = opensub_sampled[["ko", "my"]].copy()
    opensub_out["source"] = "OpenSubtitles"

    final_df = pd.concat([ted_out, opensub_out], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)  # 소스 섞기

    final_df[["ko", "my"]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    final_df.to_csv(ROOT / "data" / "myanmar_adapter" / "my_adapter_train_with_source.csv", index=False, encoding="utf-8-sig")

    report_rows = [
        {"source": "TED2020", "raw": n_ted_before, "final": len(ted_out), "pct": len(ted_out) / len(final_df)},
        {"source": "OpenSubtitles", "raw": n_opensub_before, "final": len(opensub_out), "pct": len(opensub_out) / len(final_df)},
        {"source": "bible-uedin", "raw": n_bible, "final": 0, "pct": 0.0},
    ]
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n=== 최종 결과 ===")
    print(f"총 {len(final_df)}쌍 -> {OUT_CSV.name}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
