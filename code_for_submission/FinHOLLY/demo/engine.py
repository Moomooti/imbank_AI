"""FinHOLLY 번역 엔진 — CTranslate2 + transformers 토크나이저만 있으면 동작.

의도적으로 최소 의존성만 쓴다(ctranslate2, transformers, sentencepiece) --
설치형 배포판이라 무거운 패키지(torch 등)를 끌어오지 않는다.

모델 교체: config.py가 가리키는 models/finholly_ct2_int8, models/tokenizer
폴더만 새 것으로 바꾸면 이 파일은 코드 수정 없이 새 모델을 그대로 쓴다.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ctranslate2              
from transformers import AutoTokenizer              

import config              


class TranslationEngine:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        model_dir = Path(model_dir or config.CT2_MODEL_DIR)
        tokenizer_dir = Path(tokenizer_dir or config.TOKENIZER_DIR)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"모델 폴더가 없습니다: {model_dir}\n"
                f"models 폴더 아래에 finholly_ct2_int8 폴더가 있는지 확인하세요."
            )
        if not tokenizer_dir.exists():
            raise FileNotFoundError(
                f"토크나이저 폴더가 없습니다: {tokenizer_dir}\n"
                f"models 폴더 아래에 tokenizer 폴더가 있는지 확인하세요."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=device or config.DEVICE,
            compute_type=compute_type or config.COMPUTE_TYPE,
        )

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        source_lang = source_lang or config.SOURCE_LANG
        self.tokenizer.src_lang = source_lang
        src_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

        results = self.translator.translate_batch(
            [src_tokens], target_prefix=[[target_lang]], beam_size=4
        )
        hyp = results[0].hypotheses[0]
        if hyp and hyp[0] == target_lang:                                    
            hyp = hyp[1:]
        token_ids = self.tokenizer.convert_tokens_to_ids(hyp)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def translate_all(self, text: str) -> dict[str, str]:
        return {lang: self.translate(text, lang) for lang in config.SUPPORTED_LANGS}

    def translate_batch(
        self, texts: list[str], target_lang: str, source_lang: str | None = None
    ) -> list[str]:
        """여러 문장을 한 번의 ctranslate2 호출로 번역 (크롬 확장처럼 한
        페이지에 텍스트가 많을 때, 문장마다 개별 호출하는 것보다 훨씬 빠르다).
        """
        if not texts:
            return []
        source_lang = source_lang or config.SOURCE_LANG
        self.tokenizer.src_lang = source_lang
        src_tokens_list = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t)) for t in texts
        ]
        target_prefix = [[target_lang]] * len(texts)

        results = self.translator.translate_batch(
            src_tokens_list, target_prefix=target_prefix, beam_size=4
        )
        outputs = []
        for res in results:
            hyp = res.hypotheses[0]
            if hyp and hyp[0] == target_lang:
                hyp = hyp[1:]
            token_ids = self.tokenizer.convert_tokens_to_ids(hyp)
            outputs.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return outputs


@lru_cache(maxsize=1)
def get_engine() -> TranslationEngine:
    """싱글턴 -- 서버 프로세스당 모델을 한 번만 로드한다."""
    return TranslationEngine()
