"""Phase 3.5 — mask -> translate -> restore 왕복 파이프라인.

기본은 1298개 전체, --mvp-only 주면 MVP 43개만. 용어를 3개씩 묶어 한국어
캐리어 문장을 만들고, 각 문장을:
    1) mask    : {{FINTERM001}} 형식으로 용어를 가림
    2) translate: NLLB(미얀마/태국/필리핀 경로) 또는 DeepL(베트남/인니/태국)로 번역
    3) restore : 식별자를 그 언어의 {lang}_draft 정답 번역으로 되돌림
전부 거친 뒤 생존율/위치보존율/복원성공률을 집계한다.

이 스크립트의 목적은 "번역 생산"이 아니라 mask/restore 메커니즘이 대규모에서도
깨지지 않는지 확인하는 것 — 실제 오염된 용어 찾기는 flag_suspicious_translations.py가
한다 (역할 분리).

캐리어 문장은 "본 절차는 A, B, C 등과 관련이 있습니다" 형태로 고정한다 — "등과"는
앞에 오는 용어의 받침 유무와 무관하게 항상 같은 조사형이라(등 자체가 항상
자음 받침으로 끝남), 어떤 용어를 넣어도 문법이 안 깨진다.

중간저장/이어하기: 문장 단위로 CHECKPOINT_JSON에 누적 집계를 저장하고, 재실행 시
이미 처리한 문장은 건너뛴다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv              

from app.mt.translate import translate_batch              
from app.terms import build_rows, load_merged, mask_text, restore_text              

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TARGETS = [
    ("nllb", "mya_Mymr", None),
    ("nllb", "tha_Thai", None),
    ("nllb", "tgl_Latn", None),
    ("deepl", "vie_Latn", "VI"),
    ("deepl", "ind_Latn", "ID"),
    ("deepl", "tha_Thai", "TH"),
]

CHUNK_SIZE = 3
CARRIER = "본 절차는 {terms} 등과 관련이 있습니다."
CHECKPOINT_JSON = ROOT / "data" / "masking" / "mask_pipeline_checkpoint.json"

EMPTY_AGG = {"total": 0, "survived": 0, "restored": 0, "fallback": 0, "order_ok": 0, "order_total": 0, "issues": []}


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_checkpoint() -> dict:
    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"done_sentences": [], "agg": {lang: dict(EMPTY_AGG, issues=[]) for _, lang, _ in TARGETS}}


def save_checkpoint(state: dict) -> None:
    tmp = CHECKPOINT_JSON.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(CHECKPOINT_JSON)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp-only", action="store_true", help="MVP 43개 카테고리만 (기본: 전체 1298개)")
    args = parser.parse_args()

    df = load_merged(mvp_only=args.mvp_only)
    ko_terms_all = list(zip(df["ko_term"], df["match_priority"]))
    print(f"대상 용어 수: {len(ko_terms_all)}\n")

    rows_by_lang = {
        lang: build_rows(df, lang) for lang in ["vie_Latn", "ind_Latn", "tha_Thai", "mya_Mymr", "tgl_Latn"]
    }

    sentences = []
    for chunk in chunked(ko_terms_all, CHUNK_SIZE):
        terms_str = ", ".join(t[0] for t in chunk)
        sentences.append(CARRIER.format(terms=terms_str))
    print(f"캐리어 문장 수: {len(sentences)} (문장당 최대 {CHUNK_SIZE}개 용어)\n")

    state = load_checkpoint()
    done_sentences = set(state["done_sentences"])
    agg = state["agg"]
    print(f"이어하기: 이미 완료된 문장 {len(done_sentences)}/{len(sentences)}\n")

    for sent_idx, sentence in enumerate(sentences, start=1):
        if sent_idx in done_sentences:
            continue

        masked, entries = mask_text(sentence, ko_terms_all)
        if not entries:
            done_sentences.add(sent_idx)
            continue

        for backend, lang, deepl_lang in TARGETS:
            try:
                translated = translate_batch([masked], tgt_lang=lang, backend=backend, deepl_lang=deepl_lang)[0]
            except Exception as e:                
                agg[lang]["issues"].append(f"문장{sent_idx}: 번역 API 실패 {e!r}")
                continue

            restored_text, report = restore_text(translated, entries, rows_by_lang[lang])

            a = agg[lang]
            a["total"] += report.total
            a["survived"] += report.survived
            a["restored"] += report.restored
            a["fallback"] += report.fallback_used
            a["order_ok"] += report.order_preserved_pairs
            a["order_total"] += report.order_total_pairs

            lost = [d for d in report.details if d["status"] == "lost"]
            if lost:
                a["issues"].append(f"문장{sent_idx}: 유실 {[d['ko_term'] for d in lost]} | 번역결과: {translated}")
            if report.order_total_pairs and report.order_preserved_pairs < report.order_total_pairs:
                a["issues"].append(
                    f"문장{sent_idx}: 순서 뒤집힘 ({report.order_preserved_pairs}/{report.order_total_pairs}) | 번역결과: {translated}"
                )

        done_sentences.add(sent_idx)
        state["done_sentences"] = sorted(done_sentences)
        save_checkpoint(state)

        if sent_idx % 10 == 0 or sent_idx == len(sentences):
            print(f"진행: {sent_idx}/{len(sentences)} 문장 완료 (체크포인트 저장됨)")

    print("\n" + "=" * 70)
    print("종합 리포트 (언어별)")
    print("=" * 70)
    print(f"{'언어':10s} {'생존율':>8s} {'위치보존율':>10s} {'복원성공률':>10s} {'폴백':>6s} {'이슈':>6s}")
    for _, lang, _ in TARGETS:
        a = agg[lang]
        survival = a["survived"] / a["total"] if a["total"] else float("nan")
        order = a["order_ok"] / a["order_total"] if a["order_total"] else 1.0
        restore_rate = a["restored"] / a["total"] if a["total"] else float("nan")
        print(f"{lang:10s} {survival:>7.1%} {order:>9.1%} {restore_rate:>9.1%} {a['fallback']:>6d} {len(a['issues']):>6d}")

    print()
    for _, lang, _ in TARGETS:
        if agg[lang]["issues"]:
            print(f"\n[{lang}] 이슈 상세 (최대 20건 표시):")
            for issue in agg[lang]["issues"][:20]:
                print(" -", issue)


if __name__ == "__main__":
    main()
