"""Phase 4-A 파일럿 — Gemini 문맥기반 재번역으로 오염 seed 복구 (10건).

confirmed_contamination_candidates.csv(3지표 일치 오염, 필리핀/미얀마)에서
comet_diff가 가장 나쁜 10건을 뽑는다. 각 문장에서 가장 의심스러운 용어 1개를
찾아 Context Pack(실사용 예문 + 정의문 + eng anchor + sense_id)을 만들고
Gemini로 그 용어만 문맥기반 재번역을 받는다.

검증 파이프라인 (사용자 승인기준을 파일럿 단계부터 적용):
    1) 문자/스크립트 검사 (미얀마는 유니코드 미얀마 블록 비율)
    2) 독립 계열 비교 -- Gemini 후보 vs 기존 순정 NLLB 비마스킹 전체번역 간
       LaBSE 유사도 (참고용 신호; 어절 단위 정렬이 아니라 전체 문장 대비라
       "일치/불일치"의 근사치로만 취급)
    3) held-out 재적용 -- 이 용어값만 Gemini 후보로 바꿔서 문장을 다시
       mask -> translate(NLLB) -> restore 하고, COMET-Kiwi와 LaBSE 역번역
       유사도를 "복구 전(pipeline)"/"순정 NLLB" 둘과 비교
    4) 승인 = held-out에서 COMET-Kiwi *and* LaBSE 둘 다 복구 전보다 상승
       (파일럿 간이기준: 순정 NLLB 대비 우위까지는 요구하지 않고, 상승 자체를 봄)
    5) 실패하면 abstain -- 억지로 안 채우고 기존 상태 유지 표시

산출물: pilot_gemini_repair_result.csv (10행, 판정/점수/근거 전부 기록)
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

N_PILOT = 10
TARGET_LANGS = ["tgl_Latn", "mya_Mymr"]
BACKEND = {"tgl_Latn": "nllb", "mya_Mymr": "nllb"}
MYANMAR_BLOCK = range(0x1000, 0x109F + 1)


def myanmar_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace() and unicodedata.category(c) != "Po"]
    if not chars:
        return 0.0
    in_block = sum(1 for c in chars if ord(c) in MYANMAR_BLOCK)
    return in_block / len(chars)


def pick_top10(df_confirmed: pd.DataFrame) -> pd.DataFrame:
    sub = df_confirmed[df_confirmed["lang"].isin(TARGET_LANGS)].sort_values("comet_diff")
    return sub.head(N_PILOT).reset_index(drop=True)


def find_target_term(sentence_id: str, lang: str, pipeline_df: pd.DataFrame, susp_df: pd.DataFrame) -> str | None:
    """이 문장에서 가장 의심스러운 용어 1개를 고른다."""
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


def build_context_pack(ko_term: str, ko_source: str, glossary_df: pd.DataFrame) -> dict:
    rows = glossary_df[glossary_df["ko_term"] == ko_term]
    if rows.empty:
        return {"eng": "", "ko_def": "", "sense_id": None, "n_senses": 0}
    row = rows.iloc[0]  # 다의어면 첫 sense를 기준으로(파일럿 간이처리; 다의어 자체가 섞인 후보는 없었음)
    return {
        "eng": str(row.get("eng", "") or ""),
        "ko_def": str(row.get("ko_def", "") or "")[:300],
        "sense_id": int(row["match_priority"]),
        "n_senses": len(rows),
    }


LANG_NAME = {"tgl_Latn": "필리핀어(Tagalog/Filipino)", "mya_Mymr": "미얀마어(Burmese)"}


def build_prompt(ko_term: str, ko_source: str, ctx: dict, lang: str) -> str:
    sense_note = (
        f"\n- 주의: 이 한국어 용어는 사전에 {ctx['n_senses']}개의 뜻(sense_id 포함 여러 항목)이 등록되어 있습니다. "
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


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    confirmed = pd.read_csv(ROOT / "data" / "glossary" / "confirmed_contamination_candidates.csv", encoding="utf-8-sig")
    top10 = pick_top10(confirmed)
    print(f"파일럿 대상: {len(top10)}건\n")

    susp_df = pd.read_csv(ROOT / "data" / "qa_review" / "suspicious_translations.csv", encoding="utf-8-sig")
    glossary_df = load_merged(mvp_only=False)
    ko_terms_all = list(zip(glossary_df["ko_term"], glossary_df["match_priority"]))
    rows_by_lang = {lang: build_rows(glossary_df, lang) for lang in TARGET_LANGS}
    pipeline_dfs = {lang: pd.read_csv(ROOT / "data" / "training_pairs" / f"training_pairs_{lang}.csv", encoding="utf-8-sig") for lang in TARGET_LANGS}

    results = []
    for i, cand in top10.iterrows():
        lang = cand["lang"]
        sentence_id = cand["sentence_id"]
        ko_source = cand["ko_source"]
        print(f"[{i+1}/{len(top10)}] {lang} {sentence_id}: {ko_source}")

        target_term = find_target_term(sentence_id, lang, pipeline_dfs[lang], susp_df)
        if not target_term:
            results.append({**cand.to_dict(), "verdict": "abstain", "reason": "대상 용어 특정 실패"})
            print("  -> abstain (대상 용어 특정 실패)")
            continue

        ctx = build_context_pack(target_term, ko_source, glossary_df)
        prompt = build_prompt(target_term, ko_source, ctx, lang)

        try:
            candidate_text, provider = llm_generate(prompt)
            candidate_text = candidate_text.strip().strip('"').strip("'")
        except Exception as e:  # noqa: BLE001
            results.append({**cand.to_dict(), "target_term": target_term, "verdict": "abstain", "reason": f"LLM 실패: {e!r}"})
            print(f"  -> abstain (LLM 실패: {e!r})")
            continue

        # 1) 문자/스크립트 검사
        if lang == "mya_Mymr":
            ratio = myanmar_char_ratio(candidate_text)
            if ratio < 0.5:
                results.append(
                    {**cand.to_dict(), "target_term": target_term, "candidate": candidate_text, "provider": provider,
                     "verdict": "abstain", "reason": f"미얀마 문자비율 낮음({ratio:.2f})"}
                )
                print(f"  -> abstain (미얀마 문자비율 낮음: {ratio:.2f})")
                continue
        if not candidate_text or len(candidate_text) > 200:
            results.append(
                {**cand.to_dict(), "target_term": target_term, "candidate": candidate_text, "provider": provider,
                 "verdict": "abstain", "reason": "출력 비정상(빈 값/과도한 길이)"}
            )
            print("  -> abstain (출력 비정상)")
            continue

        # 2) 독립 계열 비교 (참고용): Gemini 후보 vs 순정 NLLB 비마스킹 전체번역
        nllb_consistency = cosine_similarity_batch([candidate_text], [cand["nllb_baseline_mt"]])[0]

        # 3) held-out 재적용: 이 용어만 Gemini 후보값으로 바꿔서 재번역
        override_rows = [
            GlossaryRow(ko_term=r.ko_term, match_priority=r.match_priority,
                        term_text=candidate_text if r.ko_term == target_term else r.term_text,
                        is_loanword=r.is_loanword)
            for r in rows_by_lang[lang]
        ]
        masked, entries = mask_text(ko_source, ko_terms_all)
        translated_masked = translate_batch([masked], tgt_lang=lang, backend=BACKEND[lang])[0]
        repaired_mt, restore_report = restore_text(translated_masked, entries, override_rows)

        repaired_comet = score_batch([ko_source], [repaired_mt])[0]
        repaired_back_ko = back_translate_to_korean([repaired_mt], source_lang=lang)[0]
        repaired_labse = cosine_similarity_batch([ko_source], [repaired_back_ko])[0]

        comet_up = repaired_comet > cand["pipeline_score"]
        labse_up = repaired_labse > cand["pipeline_labse_sim"]
        approved = comet_up and labse_up

        results.append(
            {
                **cand.to_dict(),
                "target_term": target_term,
                "candidate": candidate_text,
                "provider": provider,
                "sense_id": ctx["sense_id"],
                "n_senses": ctx["n_senses"],
                "nllb_consistency_labse": nllb_consistency,
                "repaired_mt": repaired_mt,
                "repaired_comet_score": repaired_comet,
                "repaired_labse_sim": repaired_labse,
                "comet_up": comet_up,
                "labse_up": labse_up,
                "verdict": "approved" if approved else "abstain",
                "reason": "" if approved else f"comet_up={comet_up}, labse_up={labse_up}",
            }
        )
        print(f"  target_term={target_term!r} candidate={candidate_text!r} ({provider})")
        print(f"  comet: {cand['pipeline_score']:.3f} -> {repaired_comet:.3f} ({'UP' if comet_up else 'down'}) | "
              f"labse: {cand['pipeline_labse_sim']:.3f} -> {repaired_labse:.3f} ({'UP' if labse_up else 'down'}) "
              f"=> {'APPROVED' if approved else 'ABSTAIN'}")

    out_df = pd.DataFrame(results)
    out_path = ROOT / "data" / "qa_review" / "pilot_gemini_repair_result.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("파일럿 요약")
    print("=" * 90)
    print(out_df["verdict"].value_counts().to_string())
    print(f"\n저장: {out_path.name}")


if __name__ == "__main__":
    main()
