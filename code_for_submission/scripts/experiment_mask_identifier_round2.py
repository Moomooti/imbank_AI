"""라운드 2 — 1차에서 100% 생존한 4개 형식(bare/underscore/curly/square)만 놓고,
실제 mask 파이프라인에서 벌어질 법한 더 가혹한 케이스로 재검증한다.

가장 현실적인 위험: 한국어 조사가 식별자 뒤에 공백 없이 그대로 붙는 경우
(원문 "선물환을" 에서 "선물환"만 식별자로 치환하면 "FINTERM001을"이 됨 —
Phase 3 매처가 실제로 이렇게 만든다). 그 외에 문장 맨 앞에 오는 경우,
식별자끼리 쉼표로 바로 붙는 경우도 같이 확인한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv              

from app.mt.translate import translate_batch              

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

FORMATS = {
    "bare": "FINTERM{:03d}",
    "underscore": "__FINTERM{:03d}__",
    "curly": "{{{{FINTERM{:03d}}}}}",
    "square": "[FINTERM{:03d}]",
}

TARGETS = [
    ("nllb", "mya_Mymr", None),
    ("nllb", "tha_Thai", None),
    ("nllb", "tgl_Latn", None),
    ("deepl", "vie_Latn", "VI"),
    ("deepl", "ind_Latn", "ID"),
    ("deepl", "tha_Thai", "TH"),
]

                                      
CASES = {
    "조사_직접결합": "{id1}을 신청한 고객은 {id2}에 동의해야 합니다.",
    "문장_맨앞": "{id1}는 은행이 매일 확인하는 {id2}와 함께 갱신됩니다.",
    "식별자_연속쉼표": "처리 순서는 {id1},{id2} 입니다.",
}


def check(fmt: str, case_template: str, backend: str, tgt_lang: str, deepl_lang: str | None):
    id1, id2 = fmt.format(1), fmt.format(2)
    text = case_template.format(id1=id1, id2=id2)
    try:
        out = translate_batch([text], tgt_lang=tgt_lang, backend=backend, deepl_lang=deepl_lang)[0]
    except Exception as e:                
        return 0, 2, f"ERROR: {e!r}", text
    ok = sum(1 for i in (id1, id2) if i in out)
    return ok, 2, out, text


def main():
    totals = {fmt: [0, 0] for fmt in FORMATS}
    for case_name, case_template in CASES.items():
        print(f"\n{'#' * 70}\n케이스: {case_name}\n{'#' * 70}")
        for fmt_name, fmt in FORMATS.items():
            print(f"\n--- 형식: {fmt_name} ({fmt.format(1)}) ---")
            for backend, tgt_lang, deepl_lang in TARGETS:
                ok, total, out, src = check(fmt, case_template, backend, tgt_lang, deepl_lang)
                totals[fmt_name][0] += ok
                totals[fmt_name][1] += total
                mark = "OK " if ok == total else ("일부" if ok else "실패")
                print(f"  [{mark}] {backend:6s} {tgt_lang:10s} 생존 {ok}/{total}  원문: {src}")
                print(f"        출력: {out}")

    print(f"\n{'=' * 70}\n라운드2 종합 생존율\n{'=' * 70}")
    for fmt_name, (ok, total) in totals.items():
        print(f"{fmt_name:12s}: {ok}/{total}  ({ok/total:.0%})")


if __name__ == "__main__":
    main()
