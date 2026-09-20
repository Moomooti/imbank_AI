"""작업 B — 정기 통계자료 "시리즈(주제)" 식별.

시점(period)만으로 짝짓기엔 위험하다는 게 실제 수집 중 확인됨: 같은 달에
자본비율/부실채권/가계대출 등 여러 통계자료가 동시에 나오기 때문에, 시점만
같으면 전혀 다른 주제끼리 잘못 묶이는 사고가 발생했다 (예: "국내은행 자본비율"
KO가 "Delinquency Rate on WD Loans" EN과 유사도 0.59로 오매칭됨).

그래서 (기관, 시리즈, 시점) 3개가 모두 같을 때만 짝짓는다. 시리즈는 FSC/FSS가
실제로 반복 발행하는 정기 통계자료 제목에서 확인된 키워드로 식별한다(2026-09-09
기준 관찰치, 자기자본비율 항목은 실제 기사와 대조해 문구 일치까지 확인함).
"""
import re

                                 
SERIES: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("capital_ratio", re.compile(r"자기자본비율|BIS\s*기준\s*자본비율"), re.compile(r"Capital Ratios?", re.I)),
    ("npl", re.compile(r"부실채권"), re.compile(r"Delinquen|NPL|Non-?performing", re.I)),
    ("household_loan", re.compile(r"가계대출"), re.compile(r"Household Loans?", re.I)),
    ("bank_earnings", re.compile(r"은행.{0,8}(영업실적|경영실적|실적)"), re.compile(r"Bank Earnings", re.I)),
    ("insurance_earnings", re.compile(r"보험.{0,10}(실적|영업실적)"), re.compile(r"Insurance Compan.{0,20}Earnings|Earnings of Insurance", re.I)),
    ("savings_bank_earnings", re.compile(r"저축은행.{0,10}(실적|영업)"), re.compile(r"Savings Banks?.{0,20}Earnings", re.I)),
    ("specialized_credit_earnings", re.compile(r"여신전문금융회사.{0,10}실적"), re.compile(r"Specialized Credit Finance", re.I)),
    ("foreign_investor", re.compile(r"외국인.{0,8}(주식|채권).{0,8}(투자|매매)"), re.compile(r"Foreign Investors?.{0,20}Stock and Bond", re.I)),
    ("corp_issuance", re.compile(r"기업.{0,8}(주식|회사채).{0,10}발행"), re.compile(r"Corporate Equity and Debt Issues", re.I)),
    ("sbl", re.compile(r"고정이하여신"), re.compile(r"\bSBLs?\b")),
]


def match_series(title: str, lang: str) -> str | None:
    for series_id, ko_re, en_re in SERIES:
        pat = ko_re if lang == "ko" else en_re
        if pat.search(title):
            return series_id
    return None
