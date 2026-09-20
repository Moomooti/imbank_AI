"""FinHOLLY 설정 — 모델 경로 및 지원 언어.

모델을 새 버전으로 교체하려면 models/finholly_ct2_int8 폴더(그리고 필요시
models/tokenizer 폴더)만 통째로 새 것으로 바꿔치기하면 된다. 이 파일이나
demo/ 안의 코드는 손댈 필요 없다 — 전부 아래 경로 상수만 참조한다.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- 모델 교체 시 이 두 폴더만 바꾸면 됨 ----
CT2_MODEL_DIR = BASE_DIR / "models" / "finholly_ct2_int8"
TOKENIZER_DIR = BASE_DIR / "models" / "tokenizer"
# --------------------------------------------

SOURCE_LANG = "kor_Hang"

# 지원 언어 (FLORES-200 코드 -> 표시 이름). 언어를 추가/제거하려면 여기만 수정.
SUPPORTED_LANGS = {
    "vie_Latn": "베트남어",
    "ind_Latn": "인도네시아어",
    "tha_Thai": "태국어",
    "tgl_Latn": "필리핀어",
    "mya_Mymr": "미얀마어",
}

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
