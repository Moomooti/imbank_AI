"""Phase 4-A 확장 — Gemini 문맥기반 재번역으로 오염 seed 복구 (166건 전체).

pilot_gemini_repair.py에서 검증한 파이프라인(Context Pack -> LLM -> 문자검사
-> held-out 재적용 -> COMET-Kiwi/LaBSE 둘 다 상승해야 승인)을 그대로 166건
전체(confirmed_contamination_candidates.csv, 필리핀+미얀마)로 확장한다.

비용 절감 cascade (사용자 지시):
    1) 문자검사 (미얀마 유니코드 비율) -- 즉시 탈락이면 이후 단계 전부 스킵
    2) Gemini 후보 vs 순정 NLLB 비마스킹 LaBSE 일치도 -- 참고 로그만, 하드 게이트는 아님
    3) held-out 재적용 + COMET-Kiwi -- 문자검사 통과분만 수행 (NLLB 재번역 1회 필요)
    4) LaBSE 역번역 재검증 -- COMET이 이미 상승 안 했으면(AND 게이트라 어차피 기각) 생략,
       COMET이 상승한 것만 마저 LaBSE 확인 (역번역 NLLB 호출 1회 절약)

중간저장/이어하기: 문장 단위로 CHECKPOINT_CSV에 append, 재실행 시 이미 처리한
sentence_id는 건너뛴다 (Gemini rate limit으로 중간에 죽어도 이어서 가능).
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.eval import cosine_similarity_batch, score_batch  # noqa: E402
from app.llm import generate as llm_generate  # noqa: E402
from app.mt.translate import back_translate_to_korean, translate_batch  # noqa: E402
from app.terms import build_rows, load_merged, mask_text, restore_text  # noqa: E402
from app.terms.glossary_index import GlossaryRow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TARGET_LANGS = ["tgl_Latn", "mya_Mymr"]
BACKEND = {"tgl_Latn": "nllb", "mya_Mymr": "nllb"}
MYANMAR_BLOCK = range(0x1000, 0x109F + 1)
CHECKPOINT_CSV = ROOT / "data" / "glossary" / "repair_seed_166_checkpoint.csv"

FIELDNAMES = [
    "sentence_id", "lang", "ko_source", "comet_diff", "labse_diff",
    "pipeline_score", "pipeline_labse_sim", "nllb_baseline_mt",
    "target_term", "candidate", "provider", "sense_id", "n_senses",
    "nllb_consistency_labse", "repaired_mt", "repaired_comet_score",
    "repaired_labse_sim", "comet_up", "labse_up", "verdict", "reason",
]


def myanmar_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace() and unicodedata.category(c) != "Po"]
    if not chars:
        return 0.0
    in_block = sum(1 for c in chars if ord(c) in MYANMAR_BLOCK)
    return in_block / len(chars)


def find_target_term(sentence_id: str, lang: str, pipeline_df: pd.DataFrame, susp_df: pd.DataFrame) -> str | None:
    row = pipeline_df[pipeline_df["sentence_id"] == sentence_id]
    if row.empty:
        return None
    flagged = str(row.iloc[0]["flagged_terms"] or "")
    terms_used = str(row.iloc[0]["terms_used"] or "")
    candidates = [t for t in flagged.split(";") if t.strip()] or [t for t in terms_used.split(";") if t.strip()]
    if not candidates:
        return None
    scored = susp_df[(susp_df["lang"] == lang) & (susp_df["ko_term"].isin(candidates))]
    if scored.empty:
        return candidates[0]
    return scored.sort_values("suspicion_score", ascending=False).iloc[0]["ko_term"]


def build_context_pack(ko_term: str, glossary_df: pd.DataFrame) -> dict:
    rows = glossary_df[glossary_df["ko_term"] == ko_term]
    if rows.empty:
        return {"eng": "", "ko_def": "", "sense_id": None, "n_senses": 0}
    row = rows.iloc[0]
    return {
        "eng": str(row.get("eng", "") or ""),
        "ko_def": str(row.get("ko_def", "") or "")[:300],
        "sense_id": int(row["match_priority"]),
        "n_senses": len(rows),
    }


LANG_NAME = {"tgl_Latn": "필리핀어(Tagalog/Filipino)", "mya_Mymr": "미얀마어(Burmese)"}


def build_prompt(ko_term: str, ko_source: str, ctx: dict, lang: str) -> str:
    sense_note = (
        f"\n- 주의: 이 한국어 용어는 사전에 {ctx['n_senses']}개의 뜻이 등록되어 있습니다. "
        f"지금 다루는 것은 sense_id={ctx['sense_id']}, 아래 정의에 해당하는 뜻입니다."
        if ctx["n_senses"] > 1
        else ""
    )
    return f"""다음은 한국 금융 용어 "{ko_term}"의 문맥입니다.
- 정의: {ctx['ko_def'] or '(정의 없음)'}
- 영어 대응어(anchor): {ctx['eng'] or '(없음)'}
- 실제 사용 예문: "{ko_source}"{sense_note}

위 문맥에서 "{ko_term}"에 해당하는 {LANG_NAME[lang]} 표현을 정확히 하나만 출력하세요.
설명, 따옴표, 다른 언어 섞지 말고 해당 언어 단어/구 하나만 출력하세요."""


def load_checkpoint() -> pd.DataFrame:
    if CHECKPOINT_CSV.exists():
        return pd.read_csv(CHECKPOINT_CSV, encoding="utf-8-sig")
    return pd.DataFrame(columns=FIELDNAMES)


def append_checkpoint(row: dict) -> None:
    write_header = not CHECKPOINT_CSV.exists()
    df = pd.DataFrame([{k: row.get(k) for k in FIELDNAMES}])
    df.to_csv(CHECKPOINT_CSV, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    confirmed = pd.read_csv(ROOT / "data" / "glossary" / "confirmed_contamination_candidates.csv", encoding="utf-8-sig")
    targets = confirmed[confirmed["lang"].isin(TARGET_LANGS)].sort_values("comet_diff").reset_index(drop=True)
    print(f"전체 대상: {len(targets)}건 (필리핀 {(targets['lang']=='tgl_Latn').sum()}, "
          f"미얀마 {(targets['lang']=='mya_Mymr').sum()})")

    done = load_checkpoint()
    done_ids = set(zip(done["sentence_id"], done["lang"])) if len(done) else set()
    print(f"이어하기: 이미 처리됨 {len(done_ids)}건\n")

    susp_df = pd.read_csv(ROOT / "data" / "qa_review" / "suspicious_translations.csv", encoding="utf-8-sig")
    glossary_df = load_merged(mvp_only=False)
    ko_terms_all = list(zip(glossary_df["ko_term"], glossary_df["match_priority"]))
    rows_by_lang = {lang: build_rows(glossary_df, lang) for lang in TARGET_LANGS}
    pipeline_dfs = {lang: pd.read_csv(ROOT / "data" / "training_pairs" / f"training_pairs_{lang}.csv", encoding="utf-8-sig") for lang in TARGET_LANGS}

    n_processed = 0
    for i, cand in targets.iterrows():
        lang, sentence_id, ko_source = cand["lang"], cand["sentence_id"], cand["ko_source"]
        if (sentence_id, lang) in done_ids:
            continue

        row_out = {"sentence_id": sentence_id, "lang": lang, "ko_source": ko_source,
                   "comet_diff": cand["comet_diff"], "labse_diff": cand["labse_diff"],
                   "pipeline_score": cand["pipeline_score"], "pipeline_labse_sim": cand["pipeline_labse_sim"],
                   "nllb_baseline_mt": cand["nllb_baseline_mt"]}

        target_term = find_target_term(sentence_id, lang, pipeline_dfs[lang], susp_df)
        if not target_term:
            row_out.update(verdict="abstain", reason="대상 용어 특정 실패")
            append_checkpoint(row_out)
            n_processed += 1
            continue

        ctx = build_context_pack(target_term, glossary_df)
        prompt = build_prompt(target_term, ko_source, ctx, lang)
        row_out.update(target_term=target_term, sense_id=ctx["sense_id"], n_senses=ctx["n_senses"])

        try:
            candidate_text, provider = llm_generate(prompt)
            candidate_text = candidate_text.strip().strip('"').strip("'")
        except Exception as e:  # noqa: BLE001
            row_out.update(verdict="abstain", reason=f"LLM 실패: {e!r}")
            append_checkpoint(row_out)
            n_processed += 1
            continue
        row_out.update(candidate=candidate_text, provider=provider)

        # cascade 1: 문자검사 -- 실패하면 이후 단계(NLLB 재번역+COMET+LaBSE) 전부 스킵
        if lang == "mya_Mymr":
            ratio = myanmar_char_ratio(candidate_text)
            if ratio < 0.5:
                row_out.update(verdict="abstain", reason=f"미얀마 문자비율 낮음({ratio:.2f})")
                append_checkpoint(row_out)
                n_processed += 1
                continue
        if not candidate_text or len(candidate_text) > 200:
            row_out.update(verdict="abstain", reason="출력 비정상")
            append_checkpoint(row_out)
            n_processed += 1
            continue

        # cascade 2: 참고용 일치도 로그 (하드 게이트 아님)
        row_out["nllb_consistency_labse"] = cosine_similarity_batch([candidate_text], [cand["nllb_baseline_mt"]])[0]

        # cascade 3: held-out 재적용 + COMET (문자검사 통과분만)
        override_rows = [
            GlossaryRow(ko_term=r.ko_term, match_priority=r.match_priority,
                        term_text=candidate_text if r.ko_term == target_term else r.term_text,
                        is_loanword=r.is_loanword)
            for r in rows_by_lang[lang]
        ]
        masked, entries = mask_text(ko_source, ko_terms_all)
        translated_masked = translate_batch([masked], tgt_lang=lang, backend=BACKEND[lang])[0]
        repaired_mt, _ = restore_text(translated_masked, entries, override_rows)
        repaired_comet = score_batch([ko_source], [repaired_mt])[0]
        comet_up = repaired_comet > cand["pipeline_score"]
        row_out.update(repaired_mt=repaired_mt, repaired_comet_score=repaired_comet, comet_up=comet_up)

        # cascade 4: COMET이 이미 상승 안 했으면 AND 게이트 실패 확정 -> LaBSE 역번역 생략
        if not comet_up:
            row_out.update(labse_up=False, verdict="abstain", reason="comet_up=False (LaBSE 생략)")
            append_checkpoint(row_out)
            n_processed += 1
            print(f"[{n_processed}] {lang} {sentence_id}: {target_term!r}->{candidate_text!r} "
                  f"comet {cand['pipeline_score']:.3f}->{repaired_comet:.3f} ABSTAIN(comet down)")
            continue

        repaired_back_ko = back_translate_to_korean([repaired_mt], source_lang=lang)[0]
        repaired_labse = cosine_similarity_batch([ko_source], [repaired_back_ko])[0]
        labse_up = repaired_labse > cand["pipeline_labse_sim"]
        approved = comet_up and labse_up
        row_out.update(repaired_labse_sim=repaired_labse, labse_up=labse_up,
                        verdict="approved" if approved else "abstain",
                        reason="" if approved else f"labse_up={labse_up}")
        append_checkpoint(row_out)
        n_processed += 1
        print(f"[{n_processed}] {lang} {sentence_id}: {target_term!r}->{candidate_text!r} ({provider}) "
              f"comet {cand['pipeline_score']:.3f}->{repaired_comet:.3f} "
              f"labse {cand['pipeline_labse_sim']:.3f}->{repaired_labse:.3f} "
              f"=> {'APPROVED' if approved else 'ABSTAIN'}")

        if n_processed % 20 == 0:
            print(f"--- 진행: {len(done_ids) + n_processed}/{len(targets)} ---")

    final = load_checkpoint()
    print("\n" + "=" * 90)
    print(f"전체 완료: {len(final)}/{len(targets)}건")
    print("=" * 90)
    print(final["verdict"].value_counts().to_string())
    if "provider" in final.columns:
        print("\nprovider 분포 (approved만):")
        print(final[final["verdict"] == "approved"]["provider"].value_counts().to_string())


if __name__ == "__main__":
    main()
