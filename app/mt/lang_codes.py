"""FLORES-200 / NLLB language codes for the FinHOLLY target set.

대상 8종: 베트남 · 인니 · 태국 · 크메르 · 미얀마 · 필리핀 · 네팔 · (우즈베크=보류)
"""

SOURCE_LANG = "kor_Hang"  # 한국어

TARGET_LANGS = {
    "vi": "vie_Latn",  # 베트남어
    "id": "ind_Latn",  # 인도네시아어
    "th": "tha_Thai",  # 태국어
    "km": "khm_Khmr",  # 크메르어
    "my": "mya_Mymr",  # 미얀마어
    "tl": "tgl_Latn",  # 필리핀어(타갈로그)
    "ne": "npi_Deva",  # 네팔어
    # "uz": "uzn_Latn",  # 우즈베크어 — 보류 (Phase 4 범위 제외)
}

# Typology grouping referenced in Phase 3 / Phase 6-1 routing.
# A: 분절기 + 정규식 치환으로 충분 (고자원 vi/id는 파인튜닝 불필요)
# D: 차용어 원형 통과 (필리핀)
# B/C: 파인튜닝 필요 (네팔) — Phase 4
GROUP_A = ["vi", "id", "th", "km", "my"]
GROUP_D = ["tl"]
GROUP_BC = ["ne"]

# Phase 1 작업 A — financial_terms 용어 사전 번역에 쓰는 언어 5종.
# key: {lang}_draft / {lang}_src / {lang}_verified 컬럼명에 쓰는 짧은 코드
# flores: NLLB(CTranslate2) target_prefix 코드
# deepl: DeepL API target_lang 코드. None이면 DeepL 미지원 -> 항상 NLLB
GLOSSARY_LANGS = {
    "vie_Latn": {"flores": "vie_Latn", "deepl": "VI"},
    "ind_Latn": {"flores": "ind_Latn", "deepl": "ID"},
    "tha_Thai": {"flores": "tha_Thai", "deepl": "TH"},
    "tgl_Latn": {"flores": "tgl_Latn", "deepl": None},  # NLLB + loanword passthrough
    "mya_Mymr": {"flores": "mya_Mymr", "deepl": None},  # NLLB only (Phase 4에서 용어집 병합 예정)
}
