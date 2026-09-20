"""자동 평가 1단계 — COMET-Kiwi (참조 없는 QE 점수).

원어민/gold 평가셋이 없는 상황에서 "순정 NLLB vs 우리 마스킹 파이프라인"을
상대 비교하기 위한 도구. wmt22-cometkiwi-da는 소스+번역만 있으면 점수를
매기는 reference-free QE 모델이라 정답 번역 없이도 쓸 수 있다.

중요한 한계 (반드시 리포트에 같이 명시할 것):
    - 저자원 언어(특히 미얀마)·금융 전문용어에 대한 학습 신호가 WMT 데이터에
      거의 없어서, 점수의 "절대값"은 신뢰도가 낮다.
    - 그래서 이 모듈은 "A가 B보다 낫다/못하다" 같은 상대 비교 용도로만 쓰고,
      "이 번역은 87점이니 우수하다" 같은 절대평가로 해석하면 안 된다.
    - 미얀마 점수는 특히 참고용(reference only)으로만 취급.

모델은 HuggingFace의 게이트(gated) 리포지토리라 사전에 라이선스 동의 +
HF_TOKEN이 필요하다 (.env의 HF_TOKEN).
"""
from __future__ import annotations

import os
from functools import lru_cache

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


class CometKiwiUnavailable(RuntimeError):
    """모델을 못 받아왔거나(토큰/라이선스 문제) 로드에 실패했을 때."""


@lru_cache(maxsize=1)
def load_scorer():
    """COMET-Kiwi 체크포인트를 로드해 싱글턴으로 캐시한다 (무거운 모델이라 1회만).

    HF_TOKEN이 .env에 없으면 즉시 명확한 에러를 낸다 (조용히 익명 다운로드
    시도하다가 401로 죽는 것보다 원인을 바로 알 수 있게).
    """
    token = os.environ.get("HF_TOKEN") or None
    if not token:
        raise CometKiwiUnavailable(
            "HF_TOKEN이 .env에 없습니다. https://huggingface.co/Unbabel/wmt22-cometkiwi-da 에서 "
            "라이선스 동의 후 토큰을 발급받아 .env의 HF_TOKEN에 넣어주세요."
        )

    from comet import download_model, load_from_checkpoint

    try:
                                                    
        checkpoint_path = download_model(MODEL_NAME, local_files_only=True)
    except Exception:
        try:
            checkpoint_path = download_model(MODEL_NAME)
        except Exception as e:                
            raise CometKiwiUnavailable(f"모델 다운로드 실패: {e!r}") from e

    return load_from_checkpoint(checkpoint_path)


def score_batch(sources: list[str], translations: list[str], batch_size: int = 8, gpus: int = 0) -> list[float]:
    """(source, translation) 쌍마다 참조 없는 QE 점수(대략 0~1)를 매긴다.

    gpus=0 기본값 -- 이 프로젝트는 CPU 온프레미스 전제(Phase 2 원칙과 동일).
    """
    if len(sources) != len(translations):
        raise ValueError(f"sources({len(sources)})와 translations({len(translations)}) 길이가 다릅니다.")
    if not sources:
        return []

    scorer = load_scorer()
    data = [{"src": s, "mt": t} for s, t in zip(sources, translations)]
    output = scorer.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    return list(output.scores)
