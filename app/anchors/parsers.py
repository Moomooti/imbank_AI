"""작업 B — 기관별 게시판 파서 (FSC, FSS). BOK는 목록이 JS 렌더링이라 보류.

각 파서는 requests로 받은 raw HTML(bs4.BeautifulSoup)을 받아
list_* -> [{"title": str, "url": str, "date": "YYYY-MM-DD" | None}, ...]
detail_* -> {"title": str, "date": "YYYY-MM-DD" | None, "body": str}
형태로 정규화해서 반환한다.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANKLINES_RE.sub("\n\n", text).strip()


# ---------------------------------------------------------------- FSC -----

def fsc_list(html: str, base_url: str, lang: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    items = []
    if lang == "ko":
        for a in soup.select("div.board-list-wrap div.subject > a[href]"):
            items.append({
                "title": a.get("title") or a.get_text(strip=True),
                "url": urljoin(base_url, a["href"]),
                "date": None,  # 목록에 날짜 없음 -> 상세에서 채움
            })
    else:  # en
        for row in soup.select("li:has(> span.data)"):
            a = row.select_one("div.cont a[href]")
            date_span = row.select_one("span.data")
            if not a:
                continue
            title = a.select_one("dt")
            items.append({
                "title": title.get_text(strip=True) if title else a.get_text(strip=True),
                "url": urljoin(base_url, a["href"]),
                "date": _parse_en_date(date_span.get_text(strip=True)) if date_span else None,
            })
    return items


def fsc_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    subject = soup.select_one("div.board-view-wrap div.header div.subject")
    # 한글판은 div.body > div.cont 로 한 겹 더 감싸고, 영문판은 div.body 바로 밑에
    # 본문이 온다 -> 둘 다 잡히게 div.cont가 있으면 쓰고 없으면 div.body 자체를 씀
    body = soup.select_one("div.board-view-wrap div.body div.cont") or soup.select_one(
        "div.board-view-wrap div.body"
    )
    date = None
    inline_date_raw = None
    if subject:
        day_span = soup.select_one("div.board-view-wrap div.header div.day span")
        if day_span:
            date = day_span.get_text(strip=True)
        else:
            # 영문판은 subject 안에 날짜 span이 붙어있음: "TITLE<span>Sep 04, 2026</span>"
            inline_span = subject.select_one("span")
            if inline_span:
                inline_date_raw = inline_span.get_text(strip=True)
                date = _parse_en_date(inline_date_raw)
    title_text = subject.get_text(" ", strip=True) if subject else ""
    if inline_date_raw and inline_date_raw in title_text:
        title_text = title_text.replace(inline_date_raw, "").strip()
    return {
        "title": title_text,
        "date": date,
        "body": clean_text(body.get_text("\n")) if body else "",
    }


# ---------------------------------------------------------------- FSS -----

def fss_list(html: str, base_url: str, lang: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    items = []
    for row in soup.select("table tr:has(td.title)"):
        a = row.select_one("td.title a[href]")
        if not a:
            continue
        tds = row.select("td")
        date = None
        for td in tds:
            txt = td.get_text(strip=True)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", txt):
                date = txt
                break
        items.append({
            "title": a.get_text(strip=True),
            "url": urljoin(base_url, a["href"]),
            "date": date,
        })
    return items


def fss_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    subject = soup.select_one("h2.subject, h3.subject")
    body = soup.select_one("div.n-dbdata")
    date = None
    for dt in soup.select("dl.bd-info dt"):
        if dt.get_text(strip=True) in ("등록일", "Date"):
            dd = dt.find_next_sibling("dd")
            if dd:
                date = dd.get_text(strip=True)
            break
    return {
        "title": subject.get_text(strip=True) if subject else "",
        "date": date,
        "body": clean_text(body.get_text("\n")) if body else "",
    }


_EN_MONTHS_ABBR = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def _parse_en_date(text: str) -> str | None:
    """'Sep 04, 2026' -> '2026-09-04'"""
    m = re.match(r"([A-Za-z]{3})\w*\s+(\d{1,2}),?\s+(\d{4})", text.strip())
    if not m:
        return None
    mon, day, year = m.groups()
    mon_num = _EN_MONTHS_ABBR.get(mon[:3].title())
    if not mon_num:
        return None
    return f"{year}-{mon_num}-{int(day):02d}"


# ---------------------------------------------------------------- 레지스트리 -----

PARSERS = {
    "fsc": {"list": fsc_list, "detail": fsc_detail},
    "fss": {"list": fss_list, "detail": fss_detail},
}
