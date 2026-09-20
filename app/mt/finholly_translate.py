"""Phase 7 — 온프레미스 금융 번역 프로그램의 핵심 함수.

FinHOLLY 금융 어댑터가 병합된 CTranslate2 INT8 모델(finholly_ct2_int8/ +
finholly_merged_hf/)로 한국어 -> 5개 언어 번역을 제공한다. 완전히 로컬에서
동작하며 인터넷 연결이 필요 없다(온프레미스 배포 전제).
"""
from __future__ import annotations

from app.mt.ctranslate2_provider import get_finholly_provider
from app.mt.lang_codes import SOURCE_LANG

# 배포 대상 5개 언어 (Phase 1~7 전체를 관통한 FinHOLLY 최종 타겟셋)
SUPPORTED_LANGS: dict[str, str] = {
    "vie_Latn": "베트남어",
    "ind_Latn": "인도네시아어",
    "tha_Thai": "태국어",
    "tgl_Latn": "필리핀어",
    "mya_Mymr": "미얀마어",
}


def translate(korean_text: str, target_lang: str) -> str:
    """한국어 금융 문장을 target_lang(FLORES-200 코드)으로 번역."""
    if target_lang not in SUPPORTED_LANGS:
        raise ValueError(f"지원하지 않는 언어 코드: {target_lang!r}. 지원 목록: {list(SUPPORTED_LANGS)}")
    provider = get_finholly_provider()
    return provider.translate(korean_text, target_lang, source_lang=SOURCE_LANG)


def translate_all(korean_text: str) -> dict[str, str]:
    """한 문장을 지원하는 5개 언어 전부로 번역해서 {lang: 번역문} 딕셔너리로 반환."""
    provider = get_finholly_provider()
    return {
        lang: provider.translate(korean_text, lang, source_lang=SOURCE_LANG)
        for lang in SUPPORTED_LANGS
    }


if __name__ == "__main__":
    import sys
    import time

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    samples = [
        "이 고객은 마진콜이 발생하여 추가 증거금을 납부해야 합니다.",
        "증거금이 무엇인지 쉽게 설명해 주세요.",  # 금융 어댑터 대조학습 대상 용어(간접 -- 유사 문맥) 확인용
        "이 계좌를 해지하려면 어떻게 해야 하나요?",
    ]

    t0 = time.time()
    provider = get_finholly_provider()
    print(f"모델 로드: {time.time()-t0:.1f}s\n")

    for sample in samples:
        print(f"SRC (ko): {sample}")
        for lang, name in SUPPORTED_LANGS.items():
            start = time.perf_counter()
            out = translate(sample, lang)
            elapsed = time.perf_counter() - start
            print(f"  [{name}/{lang}] ({elapsed:.2f}s) {out}")
        print()
