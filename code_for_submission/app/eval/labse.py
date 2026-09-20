"""자동 평가 2단계 — LaBSE 의미 임베딩 (역번역 신뢰도 업그레이드).

기존 suspicious_translations.csv의 역번역 대조는 문자열 유사도
(difflib.SequenceMatcher, 사실상 문자 단위 lexical overlap)라 어순이 다르거나
동의어로 바뀌면 의미가 같아도 낮게 나온다("차량"↔"자동차" 등). LaBSE로
"원문 한국어"와 "역번역 결과" 둘 다 의미 임베딩으로 바꿔서 코사인 유사도를
쓰면, 표면 문자열이 달라도 의미가 같으면 유사도가 높게 나온다 -- 리서치
근거대로 역번역 판정의 신뢰도가 올라간다.

sentence-transformers 패키지는 huggingface-hub>=1.0/transformers>=5.0을
요구해서 unbabel-comet(<1.0/<5.0 요구)과 근본적으로 버전이 충돌한다. 그래서
sentence-transformers 래퍼 없이 transformers(AutoModel/AutoTokenizer)로
LaBSE를 직접 로드한다 -- 이미 comet용으로 고정해둔 버전 그대로 재사용 가능.

LaBSE는 109개 언어를 다루고 그 안에 한국어/베트남어/인도네시아어/태국어/
필리핀어(타갈로그)/미얀마어가 전부 포함된다(구글 공식 커버리지 목록 기준).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/LaBSE"
MAX_LENGTH = 64


class LaBSEUnavailable(RuntimeError):
    """모델 로드 실패."""


@lru_cache(maxsize=1)
def _load():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as e:                
        raise LaBSEUnavailable(f"LaBSE 로드 실패: {e!r}") from e
    model.eval()
    return tokenizer, model


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """텍스트 리스트 -> L2 정규화된 임베딩 행렬 (N, 768).

    LaBSE 공식 사용법대로 pooler_output(각 문장의 [CLS]를 dense+tanh로 투영한
    벡터)을 쓰고 L2 정규화한다 -- 그래야 내적이 곧 코사인 유사도가 된다.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    tokenizer, model = _load()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            out = model(**enc)
            emb = out.pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.numpy())
    return np.concatenate(all_embeddings, axis=0)


def cosine_similarity_batch(texts_a: list[str], texts_b: list[str], batch_size: int = 32) -> list[float]:
    """texts_a[i]와 texts_b[i] 간 코사인 유사도 (둘 다 L2 정규화됐으니 내적 = 코사인)."""
    if len(texts_a) != len(texts_b):
        raise ValueError(f"texts_a({len(texts_a)})와 texts_b({len(texts_b)}) 길이가 다릅니다.")
    if not texts_a:
        return []

    emb_a = embed(texts_a, batch_size=batch_size)
    emb_b = embed(texts_b, batch_size=batch_size)
    sims = np.sum(emb_a * emb_b, axis=1)
    return [float(s) for s in sims]
