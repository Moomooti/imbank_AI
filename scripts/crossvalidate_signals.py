"""자동 평가 2단계 — 3지표 교차검증 (COMET-Kiwi × LaBSE 역번역 임베딩 × suspicion_score).

1단계에서 "NLLB 경로(필리핀·미얀마)는 파이프라인이 순정보다 COMET-Kiwi 점수가
낮다"가 나왔는데, COMET-Kiwi는 저자원 언어에서 신뢰도가 낮아 그것만으론 확정할
수 없었다. 독립적인 두 번째 신호(LaBSE 역번역 임베딩 유사도)로 같은 200문장
세트를 다시 채점해서, 세 신호가 방향을 일치시키는지 본다:

    1) comet_diff      = COMET-Kiwi(파이프라인) - COMET-Kiwi(순정 NLLB)      [1단계 산출물]
    2) labse_diff       = LaBSE유사도(파이프라인 역번역) - LaBSE유사도(순정 역번역) [이번 신규]
    3) max_suspicion_score  (해당 문장에 쓰인 용어 중 최고 의심점수, 기존 산출물)

판정: comet_diff와 labse_diff가 같은 방향으로 유의미하게 상관되면(둘 다 파이프
라인이 나쁘다고 하면) "진짜 손해"로 확정. 엇갈리면 COMET-Kiwi 저자원 신뢰도
문제였을 가능성 -> 3단계(LLM-as-Judge)로 최종 확인 필요.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.eval import cosine_similarity_batch  # noqa: E402
from app.mt.translate import back_translate_to_korean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DETAIL_CSV = ROOT / "data" / "eval" / "cometkiwi_baseline_detail.csv"
NLLB_BATCH = 16


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    df = pd.read_csv(DETAIL_CSV, encoding="utf-8-sig")
    print(f"대상: {len(df)}행 (COMET-Kiwi baseline 200문장 x 5언어)")

    # 1) 역번역 (순정 NLLB 번역결과, 파이프라인 번역결과 각각 한국어로) -- 언어별로 묶어서 처리
    nllb_back = [None] * len(df)
    pipeline_back = [None] * len(df)
    for lang, idx in df.groupby("lang").groups.items():
        idx = list(idx)
        nllb_texts = df.loc[idx, "nllb_baseline_mt"].astype(str).tolist()
        pipeline_texts = df.loc[idx, "pipeline_mt"].astype(str).tolist()

        nllb_bt, pipeline_bt = [], []
        for chunk in chunked(nllb_texts, NLLB_BATCH):
            nllb_bt.extend(back_translate_to_korean(chunk, source_lang=lang))
        for chunk in chunked(pipeline_texts, NLLB_BATCH):
            pipeline_bt.extend(back_translate_to_korean(chunk, source_lang=lang))

        for i, row_idx in enumerate(idx):
            nllb_back[row_idx] = nllb_bt[i]
            pipeline_back[row_idx] = pipeline_bt[i]
        print(f"[{lang}] 역번역 완료 ({len(idx)}개)")

    df["nllb_back_ko"] = nllb_back
    df["pipeline_back_ko"] = pipeline_back

    # 2) LaBSE 의미 임베딩 유사도 (원문 한국어 vs 역번역 한국어)
    ko_sources = df["ko_source"].astype(str).tolist()
    df["nllb_labse_sim"] = cosine_similarity_batch(ko_sources, df["nllb_back_ko"].astype(str).tolist())
    df["pipeline_labse_sim"] = cosine_similarity_batch(ko_sources, df["pipeline_back_ko"].astype(str).tolist())
    df["labse_diff"] = df["pipeline_labse_sim"] - df["nllb_labse_sim"]
    print("LaBSE 임베딩 유사도 계산 완료")

    # 3) 기존 suspicion_score 병합 (문장에 쓰인 용어 중 최고 의심점수)
    df["max_suspicion_score"] = 0
    for lang in df["lang"].unique():
        pipeline_df = pd.read_csv(ROOT / "data" / "training_pairs" / f"training_pairs_{lang}.csv", encoding="utf-8-sig")
        susp_by_id = dict(zip(pipeline_df["sentence_id"], pipeline_df["max_suspicion_score"]))
        mask = df["lang"] == lang
        df.loc[mask, "max_suspicion_score"] = df.loc[mask, "sentence_id"].map(susp_by_id).fillna(0)

    df = df.rename(columns={"diff(pipeline-nllb)": "comet_diff"})

    out_path = ROOT / "data" / "eval" / "crossvalidate_detail.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n상세 저장: {out_path.name} ({len(df)}행)")

    # 4) 언어별 상관분석 + 3지표 합의도
    print("\n" + "=" * 100)
    print("언어별 상관관계 (comet_diff vs labse_diff, 둘 다 파이프라인-순정 비교)")
    print("=" * 100)
    corr_rows = []
    for lang in ["vie_Latn", "ind_Latn", "tha_Thai", "tgl_Latn", "mya_Mymr"]:
        sub = df[df["lang"] == lang]
        corr_comet_labse = sub["comet_diff"].corr(sub["labse_diff"])
        corr_comet_susp = sub["comet_diff"].corr(sub["max_suspicion_score"])
        corr_labse_susp = sub["labse_diff"].corr(sub["max_suspicion_score"])
        agree = ((sub["comet_diff"] < 0) == (sub["labse_diff"] < 0)).mean()
        both_bad = ((sub["comet_diff"] < 0) & (sub["labse_diff"] < 0)).sum()
        both_good = ((sub["comet_diff"] > 0) & (sub["labse_diff"] > 0)).sum()
        corr_rows.append(
            {
                "lang": lang,
                "comet_diff_mean": sub["comet_diff"].mean(),
                "labse_diff_mean": sub["labse_diff"].mean(),
                "corr(comet,labse)": corr_comet_labse,
                "corr(comet,suspicion)": corr_comet_susp,
                "corr(labse,suspicion)": corr_labse_susp,
                "direction_agree_rate": agree,
                "both_worse_n": both_bad,
                "both_better_n": both_good,
                "n": len(sub),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_path = ROOT / "data" / "eval" / "crossvalidate_summary.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(corr_df.to_string(index=False))
    print(f"\n요약 저장: {corr_path.name}")

    # 5) 세 지표 모두 "나쁘다"에 일치하는 문장 = 진짜 오염 후보
    confirmed = df[
        (df["comet_diff"] < -0.02) & (df["labse_diff"] < -0.02) & (df["max_suspicion_score"] > 0)
    ].sort_values("comet_diff")
    confirmed_path = ROOT / "data" / "glossary" / "confirmed_contamination_candidates.csv"
    confirmed.to_csv(confirmed_path, index=False, encoding="utf-8-sig")
    print(f"\n3지표 일치 오염 후보: {len(confirmed)}건 -> {confirmed_path.name}")


if __name__ == "__main__":
    main()
