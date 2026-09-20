"""Phase 3 — A/D 그룹 용어 치환기 데모 및 검증.

MVP 카테고리(계좌/이체/해외송금·외환/예금상품/카드·전자금융/금융 지원, 43개 용어)로
한정해서, 언어별 샘플 문장에 용어를 2개 이상 심어 넣고:
    1) 두 매치 모두 정확히 찾아지는지
    2) 뒤에서 앞으로 역순 치환해도 오프셋이 안 밀리는지
    3) 매치 구간 바깥의 원문이 한 글자도 안 바뀌는지 (미얀마 포함)
    4) (필리핀) is_loanword=True 용어는 그대로 통과, 아니면 치환되는지
를 assert로 검증한다.

주의: 캐리어 문장에 심는 용어 텍스트는 반드시 실제 {lang}_draft 값을 그대로
가져와 쓴다(term_text() 헬퍼) — 손으로 옮겨 적으면 NLLB가 붙인 종결부호(예:
미얀마어 뒤에 붙는 "။") 같은 걸 놓쳐서 매칭이 실패한다(실제로 한 번 이렇게
실패해서 고침).

치환 값(value_fn)은 지금은 "이 자리에 어떤 용어가 매치됐는지"를 보여주는
`[[ko_term]]` 데모용 태그다 — 실제 운영에서 무엇으로 바꿀지(사전 번역으로
정규화 / 다른 표기 등)는 아직 정해지지 않아 placeholder로 둔 것.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.terms import GlossaryRow, apply_matches, build_rows, find_term_matches, load_merged              
from app.terms.matcher import TermMatch, _resolve_overlaps              


def term_text(rows: list[GlossaryRow], ko_term: str) -> str:
    """실제 {lang}_draft 값을 그대로 가져온다 (손으로 옮겨 적다 생기는 오타 방지)."""
    return next(r.term_text for r in rows if r.ko_term == ko_term)


def demo_value(m) -> str:
    """검증용 치환값: loanword는 안 건드리고 그대로 통과, 나머지는 [[ko_term]]
    태그로 바꿔서 '어떤 용어가 어디서 매치됐는지' 눈으로 보이게 함."""
    if m.is_loanword:
        return m.matched_text
    return f"[[{m.ko_term}]]"


def run_case(label: str, lang: str, text: str, rows: list[GlossaryRow], expect_terms: set[str]) -> str:
    matches = find_term_matches(text, lang, rows)
    found_terms = {m.ko_term for m in matches}
    print(f"--- {label} ({lang}) ---")
    print("원문   :", text)
    print("매치   :", [(m.ko_term, m.matched_text, m.start, m.end, "loanword" if m.is_loanword else "") for m in matches])
    assert expect_terms <= found_terms, f"기대한 용어를 다 못 찾음: 기대={expect_terms} 실제={found_terms}"

    result = apply_matches(text, matches, demo_value)
    print("치환결과:", result)

                                                 
    ordered = sorted(matches, key=lambda m: m.start)
    cursor_orig = 0
    cursor_new = 0
    for m in ordered:
        gap_orig = text[cursor_orig : m.start]
        gap_new = result[cursor_new : cursor_new + len(gap_orig)]
        assert gap_orig == gap_new, f"매치 사이 구간이 손상됨: {gap_orig!r} != {gap_new!r}"
        cursor_orig = m.end
        cursor_new += len(gap_orig) + len(demo_value(m))
    tail_orig = text[cursor_orig:]
    tail_new = result[cursor_new:]
    assert tail_orig == tail_new, f"마지막 구간이 손상됨: {tail_orig!r} != {tail_new!r}"
    print("OK: 매치 구간 바깥 원문 100% 보존, 오프셋 안 밀림\n")
    return result


def check_overlap_resolution() -> None:
    """겹치는 매치 처리 규칙(긴 용어 먼저, 그다음 match_priority) 단위 검증.
    실제 43개 용어 번역문 중엔 서로 substring 관계인 자연스러운 예가 없어서,
    같은 구간에서 겹치는 3개의 가상 매치를 직접 구성해 _resolve_overlaps만
    떼어 검증한다."""
    candidates = [
        TermMatch(10, 20, "x" * 10, "짧은용어A", match_priority=1, is_loanword=False),                     
        TermMatch(10, 25, "x" * 15, "긴용어B", match_priority=999, is_loanword=False),                        
        TermMatch(12, 18, "x" * 6, "겹치는짧은용어C", match_priority=0, is_loanword=False),                   
        TermMatch(30, 40, "y" * 10, "안겹치는용어D", match_priority=5, is_loanword=False),             
    ]
    resolved = _resolve_overlaps(candidates)
    resolved_terms = {m.ko_term for m in resolved}
    print("겹치는 매치 후보:", [(m.ko_term, m.start, m.end) for m in candidates])
    print("채택된 매치     :", [(m.ko_term, m.start, m.end) for m in resolved])
    assert resolved_terms == {"긴용어B", "안겹치는용어D"}, f"겹침 해소 규칙 위반: {resolved_terms}"
                          
    for i, a in enumerate(resolved):
        for b in resolved[i + 1 :]:
            assert a.end <= b.start or b.end <= a.start, "채택된 매치끼리 겹침"
    print("OK: 긴 용어(15자)가 짧은 용어(10자)를 이기고, 그 짧은 용어와 겹치는 더 짧은 후보는 자동 탈락\n")


def main():
    check_overlap_resolution()

    df = load_merged(mvp_only=True)
    print(f"MVP 카테고리 용어 수: {len(df)}\n")

    rows = {lang: build_rows(df, lang) for lang in ["vie_Latn", "ind_Latn", "tha_Thai", "mya_Mymr", "tgl_Latn"]}

                                  
    t1, t2 = term_text(rows["vie_Latn"], "자동계좌이체"), term_text(rows["vie_Latn"], "사고신고")
    run_case(
        "자동계좌이체 + 사고신고",
        "vie_Latn",
        f"Khách hàng đã đăng ký {t1} và cần gửi {t2} trong vòng 24 giờ.",
        rows["vie_Latn"],
        {"자동계좌이체", "사고신고"},
    )

                                    
    t1, t2 = term_text(rows["ind_Latn"], "자동계좌이체"), term_text(rows["ind_Latn"], "사고신고")
    run_case(
        "자동계좌이체 + 사고신고",
        "ind_Latn",
        f"Nasabah mengaktifkan {t1} dan mengirimkan {t2} ke bank.",
        rows["ind_Latn"],
        {"자동계좌이체", "사고신고"},
    )

                                                       
                                               
                                                       
    t1, t2 = term_text(rows["tha_Thai"], "자동계좌이체"), term_text(rows["tha_Thai"], "사고신고")
    run_case(
        "자동계좌이체 + 사고신고 (공백 없이 밀착, custom_dict 검증)",
        "tha_Thai",
        f"ลูกค้าเปิดใช้งาน{t1}และต้องส่ง{t2}ภายใน24ชั่วโมง",
        rows["tha_Thai"],
        {"자동계좌이체", "사고신고"},
    )

                                                            
    t1, t2 = term_text(rows["mya_Mymr"], "자동계좌이체"), term_text(rows["mya_Mymr"], "사고신고")
    run_case(
        "자동계좌이체 + 사고신고",
        "mya_Mymr",
        f"ဖောက်သည်သည် {t1} ကို စတင်အသုံးပြုပြီး၊ {t2} ကို တင်ပြရပါမည်",
        rows["mya_Mymr"],
        {"자동계좌이체", "사고신고"},
    )

                                              
    t1, t2 = term_text(rows["tgl_Latn"], "CHIPS"), term_text(rows["tgl_Latn"], "양도성예금증서")
    tgl_text = f"Ginagamit ng bangko ang {t1} para sa {t2} araw-araw."
    run_case(
        "CHIPS(loanword, 통과) + 양도성예금증서(치환)",
        "tgl_Latn",
        tgl_text,
        rows["tgl_Latn"],
        {"CHIPS", "양도성예금증서"},
    )
                                 
    tgl_matches = find_term_matches(tgl_text, "tgl_Latn", rows["tgl_Latn"])
    chips_match = next(m for m in tgl_matches if m.ko_term == "CHIPS")
    assert chips_match.is_loanword, "CHIPS는 is_loanword=True 여야 함"
    assert chips_match.matched_text == t1, "loanword는 원문 그대로 매치되어야 함"
    print("OK: D그룹 loanword(CHIPS) 판정 및 원문 그대로 통과 확인\n")

    print("=" * 60)
    print("전체 검증 통과")


if __name__ == "__main__":
    main()
