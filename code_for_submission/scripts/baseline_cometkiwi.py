"""자동 평가 1단계 — COMET-Kiwi baseline: 순정 NLLB vs 우리 마스킹 파이프라인.

원어민/gold 평가셋 없이, 참조 없는 QE 점수(COMET-Kiwi, wmt22-cometkiwi-da)로
"순정 NLLB로만 번역" vs "우리 마스킹 파이프라인(Phase 3.5, training_pairs_*.csv에
이미 있는 실제 산출물)" 을 언어별로 상대 비교한다.

*** 중요 한계 (리포트에 반드시 같이 명시) ***
- COMET-Kiwi는 저자원 언어(미얀마)·금융 전문용어 도메인에 대한 학습 신호가
  WMT 데이터에 거의 없어 절대 점수의 신뢰도가 낮다.
  -> "A 점수가 87이라 우수하다" X, "A가 B보다 높다/낮다"만 O.
- 미얀마 점수는 특히 참고용(reference only).

표본: 한국어 문장셋 1,482개 중 무작위 N개(기본 200), 시드 고정(재현성 42).
참고: training_pairs_{lang}.csv는 1차 사전보강(21개) 반영 이전 스냅샷이라
(전체의 1.4%인 21개 문장만 옛 버전) 미세한 오차가 있을 수 있음 -- baseline
sanity check 목적이라 재생성 없이 진행.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd              
from dotenv import load_dotenv              

from app.eval import score_batch              
from app.mt.translate import translate_batch              

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SOURCE_CSV = ROOT / "data" / "source" / "한국어_금융문장_후보셋_최종.csv"
LANGS = ["vie_Latn", "ind_Latn", "tha_Thai", "tgl_Latn", "mya_Mymr"]
SAMPLE_SIZE = 200
SEED = 42
NLLB_BATCH = 16
COMET_BATCH = 8


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    src_df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    sample = src_df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    print(f"표본: {len(sample)}개 문장 (seed={SEED})")

    ko_sources = sample["최종사용문장"].astype(str).tolist()
    sentence_ids = sample["문장ID"].tolist()

    rows = []
    for lang in LANGS:
        print(f"\n=== {lang} ===")

                                    
        nllb_out = []
        for chunk in chunked(ko_sources, NLLB_BATCH):
            nllb_out.extend(translate_batch(chunk, tgt_lang=lang, backend="nllb"))
        print(f"  순정 NLLB 번역 완료 ({len(nllb_out)}개)")

                                                                
        pipeline_df = pd.read_csv(ROOT / "data" / "training_pairs" / f"training_pairs_{lang}.csv", encoding="utf-8-sig")
        pipeline_by_id = dict(zip(pipeline_df["sentence_id"], pipeline_df["target_sentence"]))
        pipeline_out = [pipeline_by_id.get(sid, "") for sid in sentence_ids]
        missing = sum(1 for t in pipeline_out if not t)
        if missing:
            print(f"  경고: 파이프라인 산출물에서 {missing}개 문장 못 찾음(0점 처리 안 하고 건너뜀 후보)")

                                          
        nllb_scores = []
        for chunk_src, chunk_mt in zip(chunked(ko_sources, COMET_BATCH), chunked(nllb_out, COMET_BATCH)):
            nllb_scores.extend(score_batch(chunk_src, chunk_mt, batch_size=COMET_BATCH))
        pipeline_scores = []
        for chunk_src, chunk_mt in zip(chunked(ko_sources, COMET_BATCH), chunked(pipeline_out, COMET_BATCH)):
            pipeline_scores.extend(score_batch(chunk_src, chunk_mt, batch_size=COMET_BATCH))
        print(f"  COMET-Kiwi 채점 완료")

        for sid, src, n_mt, n_sc, p_mt, p_sc in zip(
            sentence_ids, ko_sources, nllb_out, nllb_scores, pipeline_out, pipeline_scores
        ):
            rows.append(
                {
                    "lang": lang,
                    "sentence_id": sid,
                    "ko_source": src,
                    "nllb_baseline_mt": n_mt,
                    "nllb_baseline_score": n_sc,
                    "pipeline_mt": p_mt,
                    "pipeline_score": p_sc,
                    "diff(pipeline-nllb)": p_sc - n_sc,
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_path = ROOT / "data" / "eval" / "cometkiwi_baseline_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n상세 저장: {detail_path.name} ({len(detail_df)}행)")

    summary = (
        detail_df.groupby("lang")
        .agg(
            nllb_mean=("nllb_baseline_score", "mean"),
            pipeline_mean=("pipeline_score", "mean"),
            diff_mean=("diff(pipeline-nllb)", "mean"),
            pipeline_wins=("diff(pipeline-nllb)", lambda s: (s > 0).sum()),
            nllb_wins=("diff(pipeline-nllb)", lambda s: (s < 0).sum()),
            ties=("diff(pipeline-nllb)", lambda s: (s == 0).sum()),
            n=("diff(pipeline-nllb)", "count"),
        )
        .reset_index()
    )
    summary_path = ROOT / "data" / "eval" / "cometkiwi_baseline_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("COMET-Kiwi baseline 비교표 (절대점수 아님 -- 상대비교로만 해석, 미얀마는 특히 참고용)")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\n요약 저장: {summary_path.name}")


if __name__ == "__main__":
    main()
