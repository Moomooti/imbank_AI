"""Phase 2 — 분절 결과 공통 타입.

태국어/미얀마어 토크나이저는 둘 다 "조각(문자열) 리스트"만 돌려주고 원문 내
위치(offset)는 안 준다. Phase 3(용어 치환)에서 원문 문자열의 특정 구간을
바꿔치기하려면 각 토큰이 원문의 어디서 어디까지인지가 필요하므로, 여기서
오프셋을 복원해 Token으로 감싼다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str
    start: int                     
    end: int                                             


def tokens_with_offsets(text: str, pieces: list[str]) -> list[Token]:
    """토크나이저가 뱉은 조각들을 원문 순서대로 다시 찾아 오프셋을 붙인다.

    조각들이 원문에 등장하는 순서와 내용이 정확히 일치한다는 전제(공백 포함
    토크나이즈)에서 커서를 전진시키며 찾는다. 못 찾는 조각(정규화 등으로 원문과
    달라진 경우)은 조용히 버린다 — 극히 드문 경우이고, 위치 정보가 없는 토큰을
    억지로 포함시키는 것보다 안전하다.
    """
    tokens: list[Token] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        idx = text.find(piece, cursor)
        if idx == -1:
            idx = text.find(piece)                            
            if idx == -1:
                continue
        tokens.append(Token(text=piece, start=idx, end=idx + len(piece)))
        cursor = idx + len(piece)
    return tokens
