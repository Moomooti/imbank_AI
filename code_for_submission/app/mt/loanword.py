"""작업 A — 필리핀어(tgl_Latn) 처리용 loanword 판정.

기준(확정): ko_term이 ASCII/라틴 문자(숫자·공백·일부 기호 포함)로만 구성되어 있으면
영문 약어·차용어로 보고 번역하지 않는다 (eng 값을 그대로 통과, src="glossary").

예: CDS, ISIN, ANNA, ATM, back office, bear hug, bancassurance 등
실제 데이터 기준 1308행 중 250행 해당 (build_glossary.py 대상인 1298 고유 표제어 기준으로도 대부분 포함).
"""
import re

_ASCII_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-./ ]*$")


def is_loanword(ko_term: str) -> bool:
    return bool(_ASCII_TERM_RE.match(str(ko_term)))
