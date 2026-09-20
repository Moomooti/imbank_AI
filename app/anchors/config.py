"""작업 B — 수집 대상 게시판 설정.

BOK(한국은행)는 목록 페이지가 JS/AJAX 렌더링이라 단순 HTTP 수집이 안 됨
(robots.txt 자체는 /portal/, /eng/ 하위를 허용하므로 접근은 가능하나 크롤링 방식이 다름).
1차 수집 범위에서는 제외하고, FSC/FSS로 앵커가 부족하면 그때 헤드리스 브라우저
도입을 재검토하기로 함 (사용자 확인, 2026-09-09).
"""

SOURCES = {
    "fsc": {
        "name": "금융위원회",
        "parser": "fsc",
        "ko_list": "https://www.fsc.go.kr/no010101?curPage={page}",
        "en_list": "https://www.fsc.go.kr/eng/pr010101?curPage={page}",
        "base_url": "https://www.fsc.go.kr",
        "enabled": True,
    },
    "fss": {
        "name": "금융감독원",
        "parser": "fss",
        "ko_list": "https://www.fss.or.kr/fss/bbs/B0000188/list.do?menuNo=200218&pageIndex={page}",
        "en_list": "https://fss.or.kr/eng/bbs/B0000211/list.do?menuNo=400010&pageIndex={page}",
        "base_url": "https://www.fss.or.kr",
        "enabled": True,
    },
    "bok": {
        "name": "한국은행",
        "parser": None,
        "enabled": False,  # pending: 목록이 JS 렌더링, 헤드리스 브라우저 필요
        "note": "목록 페이지 JS 렌더링으로 접근 불가 (상세 페이지는 서버렌더링이라 URL만 알면 가능). "
        "playwright 등 헤드리스 브라우저 도입 시 재검토.",
    },
}
