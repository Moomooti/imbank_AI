# ============================================================================
# Phase 7 — 미얀마 언어 어댑터 LoRA 파인튜닝 (Colab Pro T4용)
#
# 분리형 아키텍처의 2번째 어댑터: "미얀마어 유창성"만 가르친다(방법1 = 독립
# 학습 -- 금융 도메인 어댑터와 서로 관여 없이 순정 NLLB 위에서 따로 학습하고,
# 나중에 조합). 그래서 이 스크립트에는 금융 도메인 어댑터 스크립트에 있던
# 대조 손실(TermGPT식 동음이의어 구별)이 없다 -- 여긴 표준 CE loss로 일반
# 번역 품질(유창성)만 올린다.
#
# 데이터는 (한국어, 미얀마어) 단일 방향 쌍(my_adapter_train.csv, 9만 개)이라
# 도메인 어댑터 스크립트에 있던 "언어별로 나눠 인코딩 후 재조립하는 collate_fn"
# 이나 "용어단위 배치 샘플러"가 필요 없다 -- 훨씬 단순한 표준 seq2seq
# 파인튜닝 구조다.
#
# 아래는 "# ===== Cell N =====" 구분선 기준으로 Colab 코드 셀 하나씩에
# 그대로 복사해 넣으면 된다.
#
# 데이터 전제: my_adapter_train.csv (컬럼: ko, my)를 Google Drive에 먼저
# 업로드해둘 것.
# ============================================================================


# ===== Cell 1: 라이브러리 설치 =====
# (이 셀만 예외적으로 Colab 매직 명령이라 느낌표 포함 그대로 붙여넣을 것)
#
# !pip install -q transformers peft accelerate sentencepiece pandas sacremoses sacrebleu
# !pip install -q --upgrade torchao
# (peft 최신판이 torchao 최신판을 요구해서 버전 충돌 남 -- 금융 도메인
#  어댑터 학습 때 겪은 문제, 미리 업그레이드해서 방지)


# ===== Cell 2: Google Drive 마운트 + 경로 설정 =====

from google.colab import drive
drive.mount('/content/drive')

import os
import torch

DRIVE_ROOT = "/content/drive/MyDrive/FinHOLLY"
DATA_DIR = f"{DRIVE_ROOT}/data"
OUT_DIR = f"{DRIVE_ROOT}/my_language_adapter_run"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = f"{DATA_DIR}/my_adapter_train.csv"
# 로컬 프로젝트 기준 원본 위치: C:\translate_1\my_adapter_train.csv (9만 행,
# ko/my 컬럼) -- 이 파일을 Drive의 FinHOLLY/data/ 밑에 그대로 업로드해서
# 경로(DRIVE_ROOT/data/my_adapter_train.csv)를 맞출 것.

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음(런타임을 GPU로 바꾸세요)")
if torch.cuda.is_available():
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


# ===== Cell 3: 데이터 로드 + train/val 분리 (9:1) =====

import pandas as pd

SRC_LANG = "kor_Hang"
TGT_LANG = "mya_Mymr"
VAL_RATIO = 0.1
SEED = 42

df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
df = df.dropna(subset=["ko", "my"]).reset_index(drop=True)
print(f"전체: {len(df)}쌍")

shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n_val = int(len(shuffled) * VAL_RATIO)
val_df = shuffled.iloc[:n_val].reset_index(drop=True)
train_df = shuffled.iloc[n_val:].reset_index(drop=True)

# ---- 첫 실행 테스트용 (끝까지 에러 없이 도는지 + 속도 확인) ----
# 성공하면 이 줄 지우고 전체 9만개로 본학습 (EPOCHS도 셀6에서 3으로 복원)
train_df = train_df.head(5000)

print(f"train: {len(train_df)}쌍, val: {len(val_df)}쌍")


# ===== Cell 4: 토크나이저/모델 로드 + LoRA 적용 =====

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

# 금융 도메인 어댑터와 동일한 T4 보수적 설정 (r=16, q/v_proj, fp16) --
# 서로 독립적으로 학습되는 별개의 LoRA 가중치라 설정을 맞춰두면 나중에
# 두 어댑터를 비교/조합할 때 일관성이 있다.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ===== Cell 5: Dataset / DataLoader (단일 언어쌍이라 표준 구조로 충분) =====

from torch.utils.data import Dataset, DataLoader


class TranslationPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.rows = df.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        return {"ko": str(r["ko"]), "my": str(r["my"])}


def make_collate_fn(tokenizer, src_lang=SRC_LANG, tgt_lang=TGT_LANG, max_length=MAX_LENGTH):
    def collate(batch):
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        sources = [b["ko"] for b in batch]
        targets = [b["my"] for b in batch]
        enc = tokenizer(
            sources, text_target=targets, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        labels = enc["labels"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


BATCH_SIZE = 16       # 대조 손실이 없어서 도메인 어댑터보다 여유 있게 잡음 (T4 보수적)
GRAD_ACCUM_STEPS = 2  # 데이터가 9만개로 충분해서 도메인 어댑터(4)보다 작게 -> 스텝 수 확보

train_ds = TranslationPairDataset(train_df)
val_ds = TranslationPairDataset(val_df)
collate_fn = make_collate_fn(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"배치 크기: {BATCH_SIZE}, epoch당 배치 수: {len(train_loader)}")


# ===== Cell 6: 학습 루프 (표준 CE loss만 -- 대조 손실 없음, 체크포인트 저장) =====

from transformers import get_linear_schedule_with_warmup

EPOCHS = 1       # 테스트용(5000개로 끝까지 도는지 + 시간 확인). 성공하면 3으로 원복
LR = 5e-4        # 금융 도메인 어댑터 실전에서 통했던 값
WARMUP_RATIO = 0.1

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
)
scaler = torch.cuda.amp.GradScaler()  # T4는 bf16 미지원 -> fp16 + GradScaler

model.train()
for epoch in range(EPOCHS):
    epoch_loss, n_batches = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        epoch_loss += out.loss.item()
        n_batches += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 500 == 0:
            print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)}  CE={epoch_loss/n_batches:.4f}")

    print(f"[epoch {epoch+1}/{EPOCHS}] 평균 CE={epoch_loss/n_batches:.4f}")

    ckpt_dir = f"{OUT_DIR}/checkpoint-epoch{epoch+1}"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  체크포인트 저장: {ckpt_dir}")

print("학습 완료")


# ===== Cell 7: 평가 -- chrF로 순정 NLLB vs 언어 어댑터 비교 (val셋, 참조 있음) =====

# my_adapter_train.csv의 my 컬럼이 곧 정답(gold) 번역이라, 참조 기반 지표인
# chrF를 바로 쓸 수 있다 (COMET-Kiwi 같은 참조 없는 모델을 따로 안 받아도 됨
# -- 더 가볍고 빠름). chrF는 문자 n-gram 기반이라 미얀마처럼 띄어쓰기 규칙이
# 느슨한 언어에서 BLEU보다 신뢰도가 높다.

import sacrebleu


@torch.no_grad()
def translate_batch(model, ko_texts, use_adapter=True, batch_size=16):
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    outputs = []
    for i in range(0, len(ko_texts), batch_size):
        batch = ko_texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        if use_adapter:
            out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_LENGTH, num_beams=4)
        else:
            with model.disable_adapter():
                out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_LENGTH, num_beams=4)
        outputs.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
    return outputs


model.eval()

# 전체 val(9천 개)은 시간이 걸리니 우선 일부만 평가 -- 필요하면 늘릴 것
EVAL_N = 500
eval_df = val_df.sample(min(EVAL_N, len(val_df)), random_state=7).reset_index(drop=True)
ko_texts = eval_df["ko"].tolist()
refs = eval_df["my"].tolist()

print(f"=== val {len(ko_texts)}개로 baseline(순정 NLLB) 평가 ===")
baseline_hyps = translate_batch(model, ko_texts, use_adapter=False)
baseline_chrf = sacrebleu.corpus_chrf(baseline_hyps, [refs])
print(f"baseline chrF: {baseline_chrf.score:.2f}")

print(f"\n=== val {len(ko_texts)}개로 언어 어댑터 평가 ===")
adapter_hyps = translate_batch(model, ko_texts, use_adapter=True)
adapter_chrf = sacrebleu.corpus_chrf(adapter_hyps, [refs])
print(f"adapter chrF: {adapter_chrf.score:.2f}")

print(f"\n=== Before vs After ===")
print(f"chrF: {baseline_chrf.score:.2f} -> {adapter_chrf.score:.2f}  ({adapter_chrf.score-baseline_chrf.score:+.2f})")

result_df = pd.DataFrame({
    "ko": ko_texts, "reference": refs,
    "baseline_mt": baseline_hyps, "adapter_mt": adapter_hyps,
})
result_df.to_csv(f"{OUT_DIR}/val_eval_result.csv", index=False, encoding="utf-8-sig")
print(f"\n상세 결과 저장: {OUT_DIR}/val_eval_result.csv")


# ===== Cell 8 (선택): COMET-Kiwi로 참조 없는 품질도 교차검증 =====
# Phase 1에서 쓴 것과 같은 방법론 -- chrF(참조 기반)와 방향이 일치하는지
# 확인하는 용도. wmt22-cometkiwi-da는 게이트 모델이라 HF 토큰 승인 필요
# (이미 로컬 프로젝트에서 승인받은 그 계정 토큰 재사용 가능).
#
# !pip install -q unbabel-comet
#
# import os
# os.environ["HF_TOKEN"] = "여기에_HF_토큰"  # 이미 gated 모델 접근 승인된 계정
# from comet import download_model, load_from_checkpoint
# comet_path = download_model("Unbabel/wmt22-cometkiwi-da")
# comet_model = load_from_checkpoint(comet_path)
#
# baseline_data = [{"src": s, "mt": h} for s, h in zip(ko_texts, baseline_hyps)]
# adapter_data = [{"src": s, "mt": h} for s, h in zip(ko_texts, adapter_hyps)]
# baseline_comet = comet_model.predict(baseline_data, batch_size=8, gpus=1).system_score
# adapter_comet = comet_model.predict(adapter_data, batch_size=8, gpus=1).system_score
# print(f"COMET-Kiwi: {baseline_comet:.4f} -> {adapter_comet:.4f} ({adapter_comet-baseline_comet:+.4f})")


# ===== Cell 9: 최종 어댑터 저장 =====

FINAL_DIR = f"{OUT_DIR}/my_language_adapter_final"
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"최종 미얀마 언어 어댑터 저장: {FINAL_DIR}")

# 다음 단계 메모 (분리형 아키텍처 조합):
# 1) 이 언어 어댑터와 금융 도메인 어댑터(domain_adapter_final)는 서로 독립
#    학습됐다(방법1) -- 추론 시 두 LoRA를 같은 베이스 위에 순차 적용하거나
#    (PEFT의 다중 어댑터 로드/스위칭, 또는 가중치 병합) 실험 필요.
# 2) 온프레미스 배포 시: 두 어댑터를 원하는 방식으로 합친 뒤
#    merge_and_unload() -> HF 포맷 저장 -> ct2-transformers-converter로
#    CTranslate2 변환 (기존 파이프라인과 동일 포맷).
# 3) 나머지 4개 언어(베트남/인니/태국/필리핀)도 같은 레시피로 복제 --
#    언어별 my_adapter_train.csv에 해당하는 데이터만 새로 큐레이션하면 됨.
