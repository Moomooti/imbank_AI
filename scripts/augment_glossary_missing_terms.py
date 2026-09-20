"""Phase 4-1 후속 — 사전 공백 51개 중 1차 배치(계좌/이체/해외송금·외환, 21개) 보강.

팀원 문장셋(1,482개)을 마스킹 파이프라인에 통과시켜 보니, 51개 문장의
'금융용어'가 우리 1,298개 전문 용어 사전에 아예 없어서(전문 용어 사전이라
"계좌이체"·"잔액" 같은 일상 은행 용어는 애초에 안 실려있었음) 마스킹 보호를
전혀 못 받았다. 사용자 지시대로 그 중 category가 계좌/이체/해외송금·외환인
21개부터(동음이의어 위험 낮은 일상 은행 용어라 seed 번역 안전) 사전에 추가한다.

추가 대상: financial_terms_categorized.csv (표제어/카테고리) +
          financial_terms_translated.csv ({lang}_draft, Task A와 동일 백엔드:
          vi/id/th=DeepL, tgl/mya=NLLB)

match_priority는 기존 최댓값(1308) 다음부터 이어서 부여해 기존 데이터와
충돌하지 않게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.mt.translate import translate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CATEGORIZED_CSV = ROOT / "data" / "glossary" / "financial_terms_categorized.csv"
TRANSLATED_CSV = ROOT / "data" / "glossary" / "financial_terms_translated.csv"

# (ko_term, eng, ko_def, category)
NEW_TERMS = [
    ("잔액", "balance", "계좌에 남아 있는 금액.", "계좌"),
    ("거래내역", "transaction history", "계좌에서 발생한 입출금 등 거래의 기록.", "계좌"),
    ("계좌번호", "account number", "은행 계좌를 식별하는 고유 번호.", "계좌"),
    ("입금", "deposit", "계좌에 돈을 넣는 것.", "계좌"),
    ("출금", "withdrawal", "계좌에서 돈을 빼는 것.", "계좌"),
    ("계좌해지", "account closure", "은행 계좌를 없애는 것.", "계좌"),
    ("계좌이체", "account transfer", "한 계좌에서 다른 계좌로 돈을 옮기는 것.", "이체"),
    ("이체한도", "transfer limit", "하루 또는 1회에 이체할 수 있는 최대 금액.", "이체"),
    ("이체수수료", "transfer fee", "계좌이체 시 부과되는 수수료.", "이체"),
    ("자동이체", "automatic transfer", "정해진 날짜에 자동으로 돈이 이체되도록 등록하는 서비스.", "이체"),
    ("예약이체", "scheduled transfer", "지정한 미래 시점에 이체가 실행되도록 예약하는 것.", "이체"),
    ("이체취소", "transfer cancellation", "실행된 이체를 취소하는 것.", "이체"),
    ("수취인", "beneficiary", "송금 또는 이체를 받는 사람.", "이체"),
    ("해외송금", "overseas remittance", "해외로 돈을 보내는 것.", "해외송금·외환"),
    ("해외송금 수수료", "overseas remittance fee", "해외송금 시 부과되는 수수료.", "해외송금·외환"),
    ("환율", "exchange rate", "한 통화를 다른 통화로 교환하는 비율.", "해외송금·외환"),
    ("SWIFT 코드", "SWIFT code", "국제 송금을 위한 은행 식별 코드.", "해외송금·외환"),
    ("수취은행", "beneficiary bank", "송금을 받는 은행.", "해외송금·외환"),
    ("송금한도", "remittance limit", "송금할 수 있는 최대 금액.", "해외송금·외환"),
    ("송금상태", "remittance status", "송금이 처리된 단계(접수/처리중/완료 등).", "해외송금·외환"),
    ("환전", "currency exchange", "한 나라의 통화를 다른 나라의 통화로 바꾸는 것.", "해외송금·외환"),
]

LANG_TARGETS = {
    "vie": ("deepl", "VI"),
    "ind": ("deepl", "ID"),
    "tha": ("deepl", "TH"),
    "tgl": ("nllb", None),
    "mya": ("nllb", None),
}
LANG_FLORES = {"vie": "vie_Latn", "ind": "ind_Latn", "tha": "tha_Thai", "tgl": "tgl_Latn", "mya": "mya_Mymr"}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    cat_df = pd.read_csv(CATEGORIZED_CSV, encoding="utf-8-sig")
    trans_df = pd.read_csv(TRANSLATED_CSV, encoding="utf-8-sig")

    existing_terms = set(cat_df["ko_term"])
    dup = [t for t, *_ in NEW_TERMS if t in existing_terms]
    if dup:
        raise SystemExit(f"이미 사전에 있는 용어가 섞여있음(중복 방지 중단): {dup}")

    start_priority = int(cat_df["match_priority"].max()) + 1
    print(f"신규 match_priority 시작값: {start_priority} (기존 최댓값: {start_priority - 1})")

    new_cat_rows = []
    new_trans_rows = []
    for i, (ko_term, eng, ko_def, category) in enumerate(NEW_TERMS):
        mp = start_priority + i
        is_short = "Y" if len(ko_term) <= 2 else None
        is_multiword = "Y" if " " in ko_term else None
        raw_term = f"{ko_term}({eng})"

        new_cat_rows.append(
            {
                "match_priority": mp,
                "ko_term": ko_term,
                "hanja": None,
                "eng": eng,
                "ko_def": ko_def,
                "is_short": is_short,
                "is_multiword": is_multiword,
                "is_dup": None,
                "raw_term": raw_term,
                "category": category,
                "related_category": None,
                "cat_confidence": "높음",
            }
        )
        new_trans_rows.append(
            {
                "match_priority": mp,
                "ko_term": ko_term,
                "hanja": None,
                "eng": eng,
                "ko_def": ko_def,
                "is_short": is_short,
                "is_multiword": is_multiword,
                "is_dup": None,
                "raw_term": raw_term,
            }
        )

    # 번역 (Task A와 동일 백엔드 조합)
    ko_terms = [t for t, *_ in NEW_TERMS]
    for prefix, (backend, deepl_lang) in LANG_TARGETS.items():
        translated = translate_batch(ko_terms, tgt_lang=LANG_FLORES[prefix], backend=backend, deepl_lang=deepl_lang)
        for row, value in zip(new_trans_rows, translated):
            row[f"{prefix}_draft"] = value
            row[f"{prefix}_src"] = backend
            row[f"{prefix}_verified"] = None
        print(f"[{prefix}] 번역 완료 ({backend})")

    cat_out = pd.concat([cat_df, pd.DataFrame(new_cat_rows)], ignore_index=True)
    trans_out = pd.concat([trans_df, pd.DataFrame(new_trans_rows)], ignore_index=True)

    # 원자적 저장 (tmp -> replace)
    for df, path in [(cat_out, CATEGORIZED_CSV), (trans_out, TRANSLATED_CSV)]:
        tmp = path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(path)

    print(f"\n저장 완료: {CATEGORIZED_CSV.name} ({len(cat_out)}행, +{len(new_cat_rows)})")
    print(f"저장 완료: {TRANSLATED_CSV.name} ({len(trans_out)}행, +{len(new_trans_rows)})")

    print("\n=== 신규 용어 번역 결과 미리보기 ===")
    preview = pd.DataFrame(new_trans_rows)
    print(preview[["ko_term", "vie_draft", "ind_draft", "tha_draft", "tgl_draft", "mya_draft"]].to_string())


if __name__ == "__main__":
    main()
