"""Phase 1 - 작업 A: financial_terms_clean.csv 에 언어별 번역 컬럼을 채워
financial_terms_translated.csv 를 생성한다.

번역은 "정답(gold)"이 아니라 "초벌 seed"다. 언어별로 3개 컬럼을 채운다:
    {lang}_draft    번역 결과
    {lang}_src      "deepl" | "nllb" | "glossary"
    {lang}_verified 검수 여부 (기본 빈값, 사람이 나중에 채움)

언어별 백엔드:
    vie_Latn / ind_Latn / tha_Thai : DeepL 우선, 키 없거나 실패하면 NLLB로 폴백
    tgl_Latn                       : is_loanword(ko_term)면 eng 그대로 통과(glossary),
                                      아니면 NLLB
    mya_Mymr                       : NLLB만 사용. merge_myanmar_official_glossary()는
                                      나중에 공식 용어집이 생기면 채워 넣을 자리(현재는 no-op)

이어하기: {lang}_draft가 이미 채워진 행은 건너뛴다. 배치마다 원자적으로 저장한다.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mt.lang_codes import GLOSSARY_LANGS  # noqa: E402
from app.mt.loanword import is_loanword  # noqa: E402
from app.mt.translate import BackendUnavailable, translate_batch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("translate_glossary")

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "glossary" / "financial_terms_clean.csv"
CSV_OUT = ROOT / "data" / "glossary" / "financial_terms_translated.csv"

DEEPL_BATCH_SIZE = 50
NLLB_BATCH_SIZE = 16


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_dataframe() -> pd.DataFrame:
    src = CSV_OUT if CSV_OUT.exists() else CSV_IN
    logger.info(f"입력 로드: {src.name} (이어하기={'예' if src == CSV_OUT else '아니오'})")
    df = pd.read_csv(src, encoding="utf-8-sig")

    for flores_lang in GLOSSARY_LANGS:
        prefix = flores_lang.split("_")[0]
        for col in (f"{prefix}_draft", f"{prefix}_src", f"{prefix}_verified"):
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna("")
    return df


def save_dataframe(df: pd.DataFrame) -> None:
    tmp = CSV_OUT.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CSV_OUT)


def probe_deepl(deepl_lang: str) -> bool:
    """이 언어에 DeepL을 쓸 수 있는지 짧은 문장으로 사전 확인."""
    try:
        translate_batch(["테스트"], tgt_lang="", backend="deepl", deepl_lang=deepl_lang)
        return True
    except Exception as e:  # noqa: BLE001 - 어떤 이유든 실패하면 NLLB로 폴백
        logger.warning(f"DeepL 사용 불가({deepl_lang}), NLLB로 폴백: {e!r}")
        return False


def translate_language(
    df: pd.DataFrame,
    flores_lang: str,
    deepl_lang: str | None,
    limit: int | None,
) -> None:
    prefix = flores_lang.split("_")[0]
    draft_col, src_col = f"{prefix}_draft", f"{prefix}_src"

    if prefix == "tgl":
        translate_tagalog(df, draft_col, src_col, limit)
        return

    mask = df[draft_col] == ""
    if not mask.any():
        logger.info(f"[{prefix}] 이미 전부 채워짐, 스킵")
        return

    backend = "nllb"
    if deepl_lang and probe_deepl(deepl_lang):
        backend = "deepl"
    logger.info(f"[{prefix}] 백엔드={backend}, 대상 행={mask.sum()}")

    unique_terms = df.loc[mask, "ko_term"].unique().tolist()
    if limit:
        unique_terms = unique_terms[:limit]

    batch_size = DEEPL_BATCH_SIZE if backend == "deepl" else NLLB_BATCH_SIZE
    cache: dict[str, str] = {}

    for i, chunk in enumerate(chunked(unique_terms, batch_size)):
        try:
            outputs = translate_batch(chunk, tgt_lang=flores_lang, backend=backend, deepl_lang=deepl_lang)
        except Exception as e:  # noqa: BLE001
            if backend == "deepl":
                logger.warning(f"[{prefix}] DeepL 배치 실패, 이후 NLLB로 폴백: {e!r}")
                backend = "nllb"
                batch_size = NLLB_BATCH_SIZE
                outputs = translate_batch(chunk, tgt_lang=flores_lang, backend=backend, deepl_lang=deepl_lang)
            else:
                raise

        cache.update(dict(zip(chunk, outputs)))

        chunk_mask = mask & df["ko_term"].isin(chunk)
        df.loc[chunk_mask, draft_col] = df.loc[chunk_mask, "ko_term"].map(cache)
        df.loc[chunk_mask, src_col] = backend

        save_dataframe(df)
        done = min((i + 1) * batch_size, len(unique_terms))
        logger.info(f"[{prefix}] {done}/{len(unique_terms)} 고유 표제어 완료 (누적 저장됨)")


def translate_tagalog(df: pd.DataFrame, draft_col: str, src_col: str, limit: int | None) -> None:
    mask = df[draft_col] == ""
    if not mask.any():
        logger.info("[tgl] 이미 전부 채워짐, 스킵")
        return

    loanword_mask = mask & df["ko_term"].apply(is_loanword)

    # 1) loanword -> eng 값 그대로 통과 (없으면 ko_term 자체를 사용: ko_term이 이미 영문인 경우가 많음)
    passthrough = df.loc[loanword_mask, "eng"].where(
        df.loc[loanword_mask, "eng"].notna() & (df.loc[loanword_mask, "eng"].str.strip() != ""),
        df.loc[loanword_mask, "ko_term"],
    )
    df.loc[loanword_mask, draft_col] = passthrough
    df.loc[loanword_mask, src_col] = "glossary"
    logger.info(f"[tgl] loanword 통과 {loanword_mask.sum()}행")
    save_dataframe(df)

    # 2) 나머지는 NLLB
    nllb_mask = mask & ~loanword_mask
    if not nllb_mask.any():
        return

    unique_terms = df.loc[nllb_mask, "ko_term"].unique().tolist()
    if limit:
        unique_terms = unique_terms[:limit]
    logger.info(f"[tgl] NLLB 대상 고유 표제어={len(unique_terms)}")

    cache: dict[str, str] = {}
    for i, chunk in enumerate(chunked(unique_terms, NLLB_BATCH_SIZE)):
        outputs = translate_batch(chunk, tgt_lang="tgl_Latn", backend="nllb")
        cache.update(dict(zip(chunk, outputs)))

        chunk_mask = nllb_mask & df["ko_term"].isin(chunk)
        df.loc[chunk_mask, draft_col] = df.loc[chunk_mask, "ko_term"].map(cache)
        df.loc[chunk_mask, src_col] = "nllb"

        save_dataframe(df)
        done = min((i + 1) * NLLB_BATCH_SIZE, len(unique_terms))
        logger.info(f"[tgl] {done}/{len(unique_terms)} 고유 표제어 완료 (누적 저장됨)")


def merge_myanmar_official_glossary(df: pd.DataFrame) -> None:
    """자리만 마련: 나중에 영어-버마어 공식 용어집이 생기면 여기서
    mya_draft/mya_src를 해당 용어에 한해 덮어쓰도록 구현한다.

    예정 형태:
        official = load_official_my_glossary(path)
        match_mask = df["eng"].isin(official) & (df["mya_src"] == "nllb")
        df.loc[match_mask, "mya_draft"] = df.loc[match_mask, "eng"].map(official)
        df.loc[match_mask, "mya_src"] = "glossary"
    """
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang",
        choices=list(GLOSSARY_LANGS) + ["all"],
        default="all",
        help="처리할 언어 (FLORES 코드). 기본 all",
    )
    parser.add_argument("--limit", type=int, default=None, help="언어당 처리할 고유 표제어 수 제한 (테스트용)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    df = load_dataframe()
    langs = list(GLOSSARY_LANGS) if args.lang == "all" else [args.lang]

    start = time.perf_counter()
    for flores_lang in langs:
        cfg = GLOSSARY_LANGS[flores_lang]
        logger.info(f"=== {flores_lang} 시작 ===")
        try:
            translate_language(df, flores_lang, cfg["deepl"], args.limit)
        except BackendUnavailable as e:
            logger.error(f"[{flores_lang}] 처리 불가: {e}")

    merge_myanmar_official_glossary(df)
    save_dataframe(df)

    elapsed = time.perf_counter() - start
    logger.info(f"완료. {CSV_OUT.name} 저장됨. 총 소요 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
