"""Phase 2 — 태국어/미얀마어 분절기 데모.

사람이 눈으로 결과를 확인하기 위한 스크립트. pytest 스타일 자동검증은 아니지만
숫자 뒤 종결부호 오탐 케이스를 assert로 명시해 회귀를 방지한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.segment import segment_words, split_myanmar_sentences              
from app.segment.myanmar import CLAUSE_SEP, SENTENCE_END              


def show_tokens(label: str, text: str, lang: str) -> None:
    tokens = segment_words(text, lang)
    print(f"[{label}] 원문: {text}")
    print(f"  토큰({len(tokens)}개): {[t.text for t in tokens]}")
    for t in tokens[:3]:
        assert text[t.start : t.end] == t.text, "오프셋 복원이 원문과 어긋남"
    print()


def main():
    print("=" * 60)
    print("태국어 단어 분절 (engine=newmm)")
    print("=" * 60)
    show_tokens(
        "가계대출 관련",
        "ธนาคารแห่งประเทศไทยประกาศขึ้นอัตราดอกเบี้ยนโยบายร้อยละ0.25",
        "tha_Thai",
    )
    show_tokens(
        "일반 문장",
        "ผมชอบกินข้าวผัดมากที่สุดในโลกเลยครับ",
        "tha_Thai",
    )

    print("=" * 60)
    print("미얀마어 단어 분절 (pyidaungsu)")
    print("=" * 60)
    show_tokens(
        "금융 관련",
        "ဗဟိုဘဏ်သည်မူဝါဒနှုန်းကိုတိုးမြှင့်ရန်ဆုံးဖြတ်ခဲ့သည်",
        "mya_Mymr",
    )

    print("=" * 60)
    print("미얀마어 문장 분절 (규칙 기반, ။만 경계로 인정)")
    print("=" * 60)

                                
    text1 = f"ကျွန်တော်မြန်မာစာ{CLAUSE_SEP}အင်္ဂလိပ်စာကို လေ့လာနေပါသည်{SENTENCE_END} နောက်တစ်နှစ်တွင် ပိုမိုကျွမ်းကျင်လာမည်{SENTENCE_END}"
    sents1 = split_myanmar_sentences(text1)
    print("입력:", text1)
    print("결과:", sents1)
    assert len(sents1) == 2, "။ 2개 -> 문장 2개여야 함"
    assert CLAUSE_SEP in sents1[0], "၊ 는 문장 안에 그대로 남아있어야 함(경계 아님)"
    print("OK: ၊는 경계로 안 쪼개짐, ။만 경계로 인정됨\n")

                                                          
    text2 = f"အပိုဒ်ခွဲများ\n၁{SENTENCE_END}ဘဏ္ဍာရေးဝန်ကြီးဌာနသည် စည်းမျဉ်းသစ်ကို ထုတ်ပြန်သည်{SENTENCE_END}\n၂{SENTENCE_END}ဘဏ်များသည် ခုနစ်ရက်အတွင်း လိုက်နာရမည်{SENTENCE_END}"
    sents2 = split_myanmar_sentences(text2)
    print("입력(번호매김 포함):", repr(text2))
    print("결과:", sents2)
    assert len(sents2) == 2, f"번호매김 ။ 2개는 경계로 안 쳐야 진짜 문장 2개만 남음 (실제: {len(sents2)}개)"
    print("OK: '၁။', '၂။' 번호매김이 문장 경계로 오탐되지 않음\n")

                                            
                                                
    text3 = f"ငွေကြေးအရေအတွက်မှာ ၁၀၀{SENTENCE_END} နောက်ထပ်စာကြောင်း{SENTENCE_END}"
    sents3 = split_myanmar_sentences(text3)
    print("입력(문장 끝이 숫자):", text3)
    print("결과:", sents3)
    assert len(sents3) == 2, "줄 중간(공백 뒤) 숫자로 끝나는 진짜 문장은 정상적으로 분리돼야 함"
    print("OK: 문장 중간의 숫자는 번호매김으로 오인되지 않음\n")


if __name__ == "__main__":
    main()
