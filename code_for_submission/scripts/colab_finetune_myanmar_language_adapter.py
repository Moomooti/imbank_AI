                                                                              
                                                
 
                                              
                                                 
                                           
                                                   
                  
 
                                                        
                                                   
                                              
           
 
                                                     
                 
 
                                                             
          
                                                                              


                              
                                              
 
                                                                                        
                                   
                                                
                                 


                                              

from google.colab import drive
drive.mount('/content/drive')

import os
import torch

DRIVE_ROOT = "/content/drive/MyDrive/FinHOLLY"
DATA_DIR = f"{DRIVE_ROOT}/data"
OUT_DIR = f"{DRIVE_ROOT}/my_language_adapter_run"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = f"{DATA_DIR}/my_adapter_train.csv"
                                                              
                                                       
                                                 

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음(런타임을 GPU로 바꾸세요)")
if torch.cuda.is_available():
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


                                                 

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

                                             
                                                
train_df = train_df.head(5000)

print(f"train: {len(train_df)}쌍, val: {len(val_df)}쌍")


                                           

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

                                                     
                                          
                          
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


BATCH_SIZE = 16                                               
GRAD_ACCUM_STEPS = 2                                             

train_ds = TranslationPairDataset(train_df)
val_ds = TranslationPairDataset(val_df)
collate_fn = make_collate_fn(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"배치 크기: {BATCH_SIZE}, epoch당 배치 수: {len(train_loader)}")


                                                               

from transformers import get_linear_schedule_with_warmup

EPOCHS = 1                                                  
LR = 5e-4                               
WARMUP_RATIO = 0.1

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
)
scaler = torch.cuda.amp.GradScaler()                                     

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


                                                     
                                                 
                                                   
                                     
 
                               
 
           
                                                               
                                                        
                                                           
                                                
 
                                                                                
                                                                              
                                                                                        
                                                                                      
                                                                                                         


                               

FINAL_DIR = f"{OUT_DIR}/my_language_adapter_final"
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"최종 미얀마 언어 어댑터 저장: {FINAL_DIR}")

                         
                                                      
                                                 
                                            
                                     
                                                                  
                                      
                                            
                                                     
