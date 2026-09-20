"""Phase 7 — LaBSE 교차언어 유사도로 깨진 정렬(딴 내용) 최종 제거.

my_adapter_candidates_pre_labse.csv(127,137쌍)에 대해 한국어/미얀마어 각각
LaBSE 임베딩을 구하고 코사인 유사도를 계산 -- 유사도가 낮으면 정렬이
깨졌거나(딴 문장) 번역 품질이 나쁜 것으로 보고 제거한다.

CPU라 전체(127K x 2 = 254K 문장) 임베딩에 시간이 걸려서 체크포인트 저장.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np              
import pandas as pd              

from app.eval import embed              

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = ROOT / "data" / "myanmar_adapter" / "my_adapter_candidates_pre_labse.csv"
CHECKPOINT_NPZ = ROOT / "labse_filter_checkpoint.npz"

BATCH_SIZE = 64


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(CANDIDATES_CSV, encoding="utf-8-sig")
    n = len(df)
    print(f"대상: {n}쌍")

    if CHECKPOINT_NPZ.exists():
        data = np.load(CHECKPOINT_NPZ)
        sims = data["sims"]
        done = int(data["done"])
        print(f"이어하기: {done}/{n} 완료된 상태에서 재개")
    else:
        sims = np.full(n, np.nan, dtype=np.float32)
        done = 0

    t0 = time.time()
    ko_texts = df["ko"].astype(str).tolist()
    my_texts = df["my"].astype(str).tolist()

    i = done
    while i < n:
        j = min(i + BATCH_SIZE, n)
        ko_emb = embed(ko_texts[i:j])
        my_emb = embed(my_texts[i:j])
        batch_sims = np.sum(ko_emb * my_emb, axis=1)
        sims[i:j] = batch_sims
        i = j

        if i % (BATCH_SIZE * 20) == 0 or i == n:
            elapsed = time.time() - t0
            rate = (i - done) / elapsed if elapsed > 0 else 0
            eta = (n - i) / rate if rate > 0 else float("inf")
            print(f"진행: {i}/{n} ({elapsed:.0f}s 경과, {rate:.1f}쌍/s, ETA {eta/60:.1f}분)")
            np.savez(CHECKPOINT_NPZ, sims=sims, done=i)

    np.savez(CHECKPOINT_NPZ, sims=sims, done=n)
    df["labse_sim"] = sims
    df.to_csv(ROOT / "data" / "myanmar_adapter" / "my_adapter_candidates_with_sim.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료. 저장: data/myanmar_adapter/my_adapter_candidates_with_sim.csv")
    print(f"유사도 분위수: {np.nanpercentile(sims, [1,5,25,50,75,95,99])}")


if __name__ == "__main__":
    main()
