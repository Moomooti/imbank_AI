"""Phase 5 — 대조 데이터 생성 (LoRA 동음이의어 구별 학습용).

confirmed 20개 용어(1등급 15 + 2등급 5)에 대해:
    1) 금융 문맥 한국어 문장 (실사용 1개 + LLM 생성 변주 2개 = 3개)
    2) 비금융 문맥 한국어 문장 (LLM 생성 3개)
    3) 각 문장을 5개 언어로 "의도된 의미"를 명시한 문맥기반 LLM 번역
    4) LaBSE로 금융/비금융 클러스터가 실제로 분리되는지 검증

Gemini/Groq 호출량이 많아(용어당 문장생성 5회 + 번역 5개언어 x 6문장 = 30회,
20개 용어 x 35회 ≈ 700회) 문장 단위로 체크포인트 저장, 재실행 시 이어감.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.eval import cosine_similarity_batch, embed  # noqa: E402
from app.llm import generate as llm_generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TARGETS_CSV = ROOT / "data" / "contrastive" / "contrastive_targets.csv"
GLOSSARY_CSV = ROOT / "data" / "glossary" / "financial_terms_v2.csv"
SENT_CSV = ROOT / "data" / "source" / "한국어_금융문장_후보셋_최종.csv"
CHECKPOINT = ROOT / "data" / "contrastive" / "contrastive_pairs_checkpoint.json"
OUT_CSV = ROOT / "data" / "contrastive" / "contrastive_pairs.csv"

LANGS = ["vie_Latn", "ind_Latn", "tha_Thai", "tgl_Latn", "mya_Mymr"]
LANG_NAME = {
    "vie_Latn": "베트남어(Vietnamese)", "ind_Latn": "인도네시아어(Indonesian)",
    "tha_Thai": "태국어(Thai)", "tgl_Latn": "필리핀어(Tagalog/Filipino)", "mya_Mymr": "미얀마어(Burmese)",
}

N_POS_GEN = 2  # 실사용 1개 + 생성 2개 = 3개
N_NEG_GEN = 3


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """일부 Groq 폴백 모델(추론형)이 <think>...</think> 사고과정을 답변에
    같이 섞어 보내는 경우가 있어, 실제 결과만 남기고 걷어낸다. 방어적 처리라
    <think> 태그가 없는 일반 응답은 그대로 통과한다."""
    return _THINK_RE.sub("", text).strip()


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"sentences": {}, "translations": {}}


def save_checkpoint(state: dict) -> None:
    tmp = CHECKPOINT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT)


def gen_sentences(term: str, sense: str, meaning_or_def: str, eng: str, n: int, real_example: str | None) -> list[str]:
    if sense == "financial":
        prompt = f"""한국 금융 용어 "{term}"을(를) 실제 금융 문맥에서 자연스럽게 사용하는 한국어 문장을 {n}개 만들어주세요.
- 정의: {meaning_or_def}
- 영어 대응어: {eng or '(없음)'}
- 참고 예문(이것과 다른 표현으로): "{real_example or ''}"
각 문장은 한 줄에 하나씩, 번호나 설명 없이 문장만 출력하세요."""
    else:
        prompt = f"""한국어 단어 "{term}"을(를) 금융과 무관한 일상적인 뜻으로 자연스럽게 사용하는 한국어 문장을 {n}개 만들어주세요.
- 이 문맥에서의 뜻: {meaning_or_def}
- 금융 용어로서의 뜻이 아니라 위에 설명한 일상적 뜻으로만 써야 합니다.
각 문장은 한 줄에 하나씩, 번호나 설명 없이 문장만 출력하세요."""
    text, provider = llm_generate(prompt)
    text = strip_thinking(text)
    lines = [ln.strip().lstrip("0123456789.-) ").strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[:n], provider


def translate_sentence(sentence: str, term: str, sense: str, meaning_or_def: str, lang: str) -> tuple[str, str]:
    sense_desc = f"금융 용어로서의 뜻: {meaning_or_def}" if sense == "financial" else f"일상적인 뜻(금융과 무관): {meaning_or_def}"
    prompt = f"""다음 한국어 문장을 {LANG_NAME[lang]}로 번역하세요.
문장: "{sentence}"
주의: 이 문장에서 "{term}"은(는) 다음 뜻으로 쓰였습니다 — {sense_desc}
이 뜻에 맞게 정확히 번역하세요. 번역문만 출력하고 다른 설명은 하지 마세요."""
    text, provider = llm_generate(prompt)
    text = strip_thinking(text)
    return text.strip().strip('"').strip("'"), provider


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    targets = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig")
    glossary = pd.read_csv(GLOSSARY_CSV, encoding="utf-8-sig")
    sent_df = pd.read_csv(SENT_CSV, encoding="utf-8-sig")

    state = load_checkpoint()

    for t in targets.itertuples(index=False):
        term = t.ko_term
        if term in state["sentences"]:
            continue
        print(f"\n=== {term} ({t.grade}) ===")

        g = glossary[glossary["ko_term"] == term]
        ko_def = str(g.iloc[0]["ko_def"])[:250] if len(g) and pd.notna(g.iloc[0]["ko_def"]) else t.nonfinancial_meaning
        eng = str(g.iloc[0]["eng"]) if len(g) and pd.notna(g.iloc[0]["eng"]) else ""

        real_rows = sent_df[sent_df["금융용어"] == term]
        real_example = str(real_rows.iloc[0]["최종사용문장"]) if len(real_rows) else None

        pos_sentences = [real_example] if real_example else []
        try:
            gen_pos, prov1 = gen_sentences(term, "financial", ko_def, eng, N_POS_GEN, real_example)
            pos_sentences += gen_pos
        except Exception as e:  # noqa: BLE001
            print(f"  긍정 생성 실패: {e!r}")

        try:
            neg_sentences, prov2 = gen_sentences(term, "nonfinancial", t.nonfinancial_meaning, "", N_NEG_GEN, None)
        except Exception as e:  # noqa: BLE001
            print(f"  부정 생성 실패: {e!r}")
            neg_sentences = []

        state["sentences"][term] = {
            "grade": t.grade,
            "ko_def": ko_def,
            "eng": eng,
            "nonfinancial_meaning": t.nonfinancial_meaning,
            "pos": pos_sentences,
            "neg": neg_sentences,
        }
        save_checkpoint(state)
        print(f"  긍정 {len(pos_sentences)}개, 부정 {len(neg_sentences)}개 문장 생성됨")

    # 번역 단계
    total_calls = sum(len(v["pos"]) + len(v["neg"]) for v in state["sentences"].values()) * len(LANGS)
    print(f"\n=== 번역 단계 (예상 {total_calls}회 호출) ===")
    done = 0
    for term, data in state["sentences"].items():
        for sense, sentences in [("financial", data["pos"]), ("nonfinancial", data["neg"])]:
            meaning = data["ko_def"] if sense == "financial" else data["nonfinancial_meaning"]
            for si, sentence in enumerate(sentences):
                for lang in LANGS:
                    key = f"{term}|{sense}|{si}|{lang}"
                    # 이전에 실패해서 translation=None으로 캐시된 것은 재시도
                    # 대상 -- 성공한 것만 스킵한다 (Groq TPD 소진으로 대량
                    # 실패했던 2026-09-10 사고 이후 수정).
                    if key in state["translations"] and state["translations"][key].get("translation") is not None:
                        continue
                    try:
                        translated, provider = translate_sentence(sentence, term, sense, meaning, lang)
                        state["translations"][key] = {"translation": translated, "provider": provider}
                    except Exception as e:  # noqa: BLE001
                        state["translations"][key] = {"translation": None, "provider": None, "error": str(e)}
                    done += 1
                    if done % 20 == 0:
                        save_checkpoint(state)
                        print(f"  번역 진행: {done}/{total_calls}")
    save_checkpoint(state)
    print(f"번역 완료: {done}/{total_calls}")

    # ---- LaBSE 품질 필터 ----
    print("\n=== LaBSE 품질 필터 ===")
    rows = []
    filter_report = []
    for term, data in state["sentences"].items():
        fin_ref = data["ko_def"]
        nonfin_ref = data["nonfinancial_meaning"]
        pos_sents, neg_sents = data["pos"], data["neg"]
        all_ko = pos_sents + neg_sents
        if not all_ko:
            continue
        refs = [fin_ref, nonfin_ref]
        emb_all = embed(all_ko + refs)
        emb_sents, emb_fin_ref, emb_nonfin_ref = emb_all[: len(all_ko)], emb_all[-2], emb_all[-1]

        import numpy as np
        sim_to_fin = emb_sents @ emb_fin_ref
        sim_to_nonfin = emb_sents @ emb_nonfin_ref

        n_pos = len(pos_sents)
        pos_ok = [sim_to_fin[i] > sim_to_nonfin[i] for i in range(n_pos)]
        neg_ok = [sim_to_nonfin[n_pos + i] > sim_to_fin[n_pos + i] for i in range(len(neg_sents))]

        margin = (sim_to_fin[:n_pos].mean() - sim_to_nonfin[:n_pos].mean() if n_pos else 0) + \
                 (sim_to_nonfin[n_pos:].mean() - sim_to_fin[n_pos:].mean() if neg_sents else 0)
        filter_report.append({"ko_term": term, "margin": float(margin),
                               "pos_pass": sum(pos_ok), "pos_total": n_pos,
                               "neg_pass": sum(neg_ok), "neg_total": len(neg_sents)})

        for i, sent in enumerate(pos_sents):
            passed = pos_ok[i]
            for lang in LANGS:
                key = f"{term}|financial|{i}|{lang}"
                tr = state["translations"].get(key, {})
                rows.append({"term": term, "grade": data["grade"], "sense": "financial", "ko_sentence": sent,
                             "lang": lang, "translation": tr.get("translation"), "label": "pos",
                             "provider": tr.get("provider"), "labse_filter_pass": passed})
        for i, sent in enumerate(neg_sents):
            passed = neg_ok[i]
            for lang in LANGS:
                key = f"{term}|nonfinancial|{i}|{lang}"
                tr = state["translations"].get(key, {})
                rows.append({"term": term, "grade": data["grade"], "sense": "nonfinancial", "ko_sentence": sent,
                             "lang": lang, "translation": tr.get("translation"), "label": "neg",
                             "provider": tr.get("provider"), "labse_filter_pass": passed})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    report_df = pd.DataFrame(filter_report)
    report_df.to_csv(ROOT / "data" / "contrastive" / "contrastive_filter_report.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장: {OUT_CSV.name} ({len(out_df)}행)")
    print(f"저장: contrastive_filter_report.csv ({len(report_df)}행)")
    print(f"\n번역 실패(translation=None): {out_df['translation'].isna().sum()}건")
    print(f"LaBSE 필터 통과: {out_df['labse_filter_pass'].sum()}/{len(out_df)}행")
    weak = report_df[report_df["margin"] < 0.05]
    print(f"대조 효과 약한 용어(margin<0.05): {len(weak)}개 -> {weak['ko_term'].tolist()}")


if __name__ == "__main__":
    main()
