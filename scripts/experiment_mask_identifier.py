"""Phase 3.5 실험 — mask/restore용 식별자 형식이 실제 번역기를 거치고도
살아남는지 검증.

같은 한국어 캐리어 문장에 후보 형식별로 식별자 2개를 심어서 NLLB(미얀마/태국/
필리핀 경로)와 DeepL(베트남/인니/태국)에 실제로 통과시킨 뒤, 출력 문자열에
식별자가 "정확히 그대로" 들어있는지 센다.

후보 형식:
    bare       FINTERM001                (구분자 없음)
    underscore __FINTERM001__            (i18n 스타일)
    curly      {{FINTERM001}}            (i18n 플레이스홀더 스타일)
    square     [FINTERM001]              (흔한 플레이스홀더)
    dbrkt      ⟦FINTERM001⟧              (희귀 유니코드 괄호, U+27E6/27E7)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from app.mt.translate import translate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

FORMATS = {
    "bare": "FINTERM{:03d}",
    "underscore": "__FINTERM{:03d}__",
    "curly": "{{{{FINTERM{:03d}}}}}",  # {{FINTERM001}}
    "square": "[FINTERM{:03d}]",
    "dbrkt": "⟦FINTERM{:03d}⟧",  # ⟦FINTERM001⟧
}

# (backend, tgt_lang(FLORES), deepl_lang)
TARGETS = [
    ("nllb", "mya_Mymr", None),
    ("nllb", "tha_Thai", None),
    ("nllb", "tgl_Latn", None),
    ("deepl", "vie_Latn", "VI"),
    ("deepl", "ind_Latn", "ID"),
    ("deepl", "tha_Thai", "TH"),
]

CARRIER = "은행은 {id1} 절차와 {id2} 규정을 동시에 적용합니다."


def check(fmt_name: str, fmt: str, backend: str, tgt_lang: str, deepl_lang: str | None):
    id1, id2 = fmt.format(1), fmt.format(2)
    text = CARRIER.format(id1=id1, id2=id2)
    try:
        out = translate_batch([text], tgt_lang=tgt_lang, backend=backend, deepl_lang=deepl_lang)[0]
    except Exception as e:  # noqa: BLE001
        return {"fmt": fmt_name, "backend": backend, "lang": tgt_lang, "ok": 0, "total": 2, "out": f"ERROR: {e!r}"}

    ok = sum(1 for i in (id1, id2) if i in out)
    return {"fmt": fmt_name, "backend": backend, "lang": tgt_lang, "ok": ok, "total": 2, "out": out}


def main():
    results = []
    for fmt_name, fmt in FORMATS.items():
        print(f"\n{'=' * 70}\n형식: {fmt_name}  예시: {fmt.format(1)}\n{'=' * 70}")
        for backend, tgt_lang, deepl_lang in TARGETS:
            r = check(fmt_name, fmt, backend, tgt_lang, deepl_lang)
            results.append(r)
            mark = "OK " if r["ok"] == r["total"] else ("일부" if r["ok"] else "실패")
            print(f"[{mark}] {backend:6s} {tgt_lang:10s} 생존 {r['ok']}/{r['total']}  출력: {r['out']}")

    print(f"\n{'=' * 70}\n요약 (형식별 총 생존율)\n{'=' * 70}")
    for fmt_name in FORMATS:
        fmt_results = [r for r in results if r["fmt"] == fmt_name]
        ok = sum(r["ok"] for r in fmt_results)
        total = sum(r["total"] for r in fmt_results)
        print(f"{fmt_name:12s}: {ok}/{total}  ({ok/total:.0%})")


if __name__ == "__main__":
    main()
