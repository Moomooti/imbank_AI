"""Phase 0 — CTranslate2 엔진 세팅.

모델(두뇌)은 NLLB 그대로, 구동 엔진만 CTranslate2로 교체하여 CPU 추론 속도를
3~4배 높인다. 토크나이저는 transformers.AutoTokenizer를 그대로 사용하고,
모델 실행부(순전파/디코딩)만 CTranslate2 Translator로 교체한다.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

from app.mt.lang_codes import SOURCE_LANG

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "nllb-200-distilled-1.3B-ct2-int8"


class CTranslate2Provider:
    """NLLB-200(distilled 1.3B) 계열 추론을 CTranslate2 엔진으로 수행하는 provider.

    Phase 0(순정 NLLB)과 Phase 7(FinHOLLY 금융 어댑터 병합 모델)이 모델
    디렉터리 구조가 다르다 -- 순정 NLLB는 토크나이저 파일이 모델과 같은
    디렉터리에 있지만, LoRA 병합 후 ct2 변환된 FinHOLLY 모델은 ct2 변환
    산출물(model.bin 등)과 HF 토크나이저 산출물(tokenizer.json 등)이
    서로 다른 디렉터리에 나뉘어 있다. tokenizer_dir을 별도로 주면 그쪽에서
    토크나이저를 읽고, 안 주면 기존처럼 model_dir에서 읽는다(하위 호환).
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        tokenizer_dir: str | Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"모델 디렉터리가 없습니다: {model_dir}. "
                "먼저 scripts/download_model.py 를 실행하세요."
            )
        tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else model_dir
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"토크나이저 디렉터리가 없습니다: {tokenizer_dir}")

        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.translator = ctranslate2.Translator(
            str(model_dir), device=device, compute_type=compute_type
        )

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = SOURCE_LANG,
        beam_size: int = 4,
    ) -> str:
        """단일 문장 번역. target_lang은 FLORES-200 코드 (예: 'vie_Latn')."""
        return self.translate_batch([text], target_lang, source_lang, beam_size)[0]

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = SOURCE_LANG,
        beam_size: int = 4,
    ) -> list[str]:
        self.tokenizer.src_lang = source_lang
        source_sents = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            for t in texts
        ]
        target_prefix = [[target_lang]] * len(texts)

        results = self.translator.translate_batch(
            source_sents,
            target_prefix=target_prefix,
            beam_size=beam_size,
        )

        outputs = []
        for res in results:
            hyp = res.hypotheses[0]
            if hyp and hyp[0] == target_lang:  # target_prefix로 넣은 언어 태그가 맨 앞에 나오면 제거
                hyp = hyp[1:]
            token_ids = self.tokenizer.convert_tokens_to_ids(hyp)
            outputs.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return outputs


@lru_cache(maxsize=1)
def get_default_provider() -> CTranslate2Provider:
    """싱글턴 provider (반복 로딩 방지) -- 순정 NLLB (Phase 0)."""
    return CTranslate2Provider()


DEFAULT_FINHOLLY_CT2_DIR = Path(__file__).resolve().parents[2] / "finholly_ct2_int8"
DEFAULT_FINHOLLY_TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "finholly_merged_hf"


@lru_cache(maxsize=1)
def get_finholly_provider() -> CTranslate2Provider:
    """싱글턴 provider -- Phase 7 FinHOLLY 금융 어댑터 병합 모델(온프레미스 배포용).

    finholly_ct2_int8/(ct2 변환 산출물)과 finholly_merged_hf/(HF 토크나이저
    산출물)이 서로 다른 디렉터리에 있어서 tokenizer_dir을 명시적으로 넘긴다.
    """
    return CTranslate2Provider(
        model_dir=DEFAULT_FINHOLLY_CT2_DIR,
        tokenizer_dir=DEFAULT_FINHOLLY_TOKENIZER_DIR,
    )


if __name__ == "__main__":
    from app.mt.lang_codes import TARGET_LANGS

    provider = CTranslate2Provider()
    sample = "이 고객은 마진콜이 발생하여 추가 증거금을 납부해야 합니다."

    print(f"SRC (ko): {sample}\n")
    for name, code in TARGET_LANGS.items():
        start = time.perf_counter()
        out = provider.translate(sample, code)
        elapsed = time.perf_counter() - start
        print(f"[{name} / {code}] ({elapsed:.2f}s)\n  {out}\n")
