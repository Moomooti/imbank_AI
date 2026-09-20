                                                                              
                                                
 
                                           
                                                
 
                                            
                                            
                                           
                       
 
                                                     
                                           
                                                       
 
                                                                            
                                                                  
                                                         
                                                                              


                              
                                              
 
                                                                              


                                              

from google.colab import drive
drive.mount('/content/drive')

import os
import torch

DRIVE_ROOT = "/content/drive/MyDrive/finholly"                   
DATA_DIR = f"{DRIVE_ROOT}/data"
OUT_DIR = f"{DRIVE_ROOT}/domain_adapter_run"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = f"{DATA_DIR}/contrastive_train.csv"
VAL_CSV = f"{DATA_DIR}/contrastive_val.csv"
TEST_CSV = f"{DATA_DIR}/contrastive_test.csv"
                                
                                                                                   

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음(런타임을 GPU로 바꾸세요)")
if torch.cuda.is_available():
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


                                 

import pandas as pd

train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")
test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

print(f"train: {len(train_df)}행, {train_df['term'].nunique()}개 용어")
print(f"val:   {len(val_df)}행, {val_df['term'].nunique()}개 용어 -> {sorted(val_df['term'].unique())}")
print(f"test:  {len(test_df)}행, {test_df['term'].nunique()}개 용어 -> {sorted(test_df['term'].unique())}")

                                                        
overlap_val = set(train_df["term"]) & set(val_df["term"])
overlap_test = set(train_df["term"]) & set(test_df["term"])
assert not overlap_val, f"val 용어가 train에도 있음: {overlap_val}"
assert not overlap_test, f"test 용어가 train에도 있음: {overlap_test}"
print("term 분리 확인 OK -- val/test 용어는 train에 전혀 없음")


                                           

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "kor_Hang"
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


                                                        

import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler


class ContrastivePairsDataset(Dataset):
    """한 행 = (한국어 원문, 목표언어, 정답 번역, term, sense label)."""

    def __init__(self, df: pd.DataFrame):
        self.rows = df.reset_index(drop=True)
        self.term_to_id = {t: i for i, t in enumerate(sorted(self.rows["term"].unique()))}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        return {
            "ko_sentence": str(r["ko_sentence"]),
            "translation": str(r["translation"]),
            "lang": r["lang"],
            "term": r["term"],
            "term_id": self.term_to_id[r["term"]],
            "label": 1 if r["label"] == "pos" else 0,
        }


class TermGroupedBatchSampler(Sampler):
    """배치마다 용어 몇 개를 고르고 그 용어의 예문들을 몰아 담는다 --
    대조 손실이 매 배치마다 실제로 비교할 대상(같은 용어의 pos/neg)을
    갖도록 보장하기 위함. 안 이러면 무작위 배치엔 같은 용어가 거의 안 겹쳐서
    대조 손실이 대부분 스킵된다."""

    def __init__(self, dataset: ContrastivePairsDataset, terms_per_batch=2, samples_per_term=4, seed=42):
        self.terms_per_batch = terms_per_batch
        self.samples_per_term = samples_per_term
        self.seed = seed
        self.epoch = 0
        self.term_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            self.term_to_indices[dataset[i]["term"]].append(i)
        self.terms = list(self.term_to_indices.keys())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        terms = self.terms[:]
        rng.shuffle(terms)
        n_batches = max(1, len(terms) // self.terms_per_batch)
        for b in range(n_batches):
            batch_terms = terms[b * self.terms_per_batch : (b + 1) * self.terms_per_batch]
            batch_idx = []
            for t in batch_terms:
                idxs = self.term_to_indices[t][:]
                rng.shuffle(idxs)
                reps = (self.samples_per_term // max(len(idxs), 1)) + 1
                batch_idx.extend((idxs * reps)[: self.samples_per_term])
            rng.shuffle(batch_idx)
            yield batch_idx

    def __len__(self):
        return max(1, len(self.terms) // self.terms_per_batch)


def make_collate_fn(tokenizer, src_lang=SRC_LANG, max_length=MAX_LENGTH):
    def collate(batch):
        tokenizer.src_lang = src_lang
        sources = [b["ko_sentence"] for b in batch]
        enc = tokenizer(sources, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

                                                         
        labels = [None] * len(batch)
        by_lang = defaultdict(list)
        for i, b in enumerate(batch):
            by_lang[b["lang"]].append(i)
        for lang, idxs in by_lang.items():
            tokenizer.tgt_lang = lang
            texts = [batch[i]["translation"] for i in idxs]
                                                                    
                                                         
            lab = tokenizer(text_target=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            for j, i in enumerate(idxs):
                labels[i] = lab["input_ids"][j]
                                          
        max_len = max(l.size(0) for l in labels)
        pad_id = tokenizer.pad_token_id
        padded = torch.full((len(labels), max_len), pad_id, dtype=torch.long)
        for i, l in enumerate(labels):
            padded[i, : l.size(0)] = l
        padded[padded == pad_id] = -100                

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": padded,
            "term_id": torch.tensor([b["term_id"] for b in batch], dtype=torch.long),
            "sense_label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }

    return collate


train_ds = ContrastivePairsDataset(train_df)
val_ds = ContrastivePairsDataset(val_df)

TERMS_PER_BATCH = 2
SAMPLES_PER_TERM = 4                                                           
GRAD_ACCUM_STEPS = 1              

train_sampler = TermGroupedBatchSampler(train_ds, TERMS_PER_BATCH, SAMPLES_PER_TERM, seed=42)
collate_fn = make_collate_fn(tokenizer)
train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn)

print(f"배치 크기: {TERMS_PER_BATCH * SAMPLES_PER_TERM}, epoch당 배치 수: {len(train_sampler)}")


                                                          

import torch.nn.functional as F


def encoder_pooled_embeddings(model, input_ids, attention_mask):
    """인코더의 마지막 hidden state를 attention_mask로 평균 풀링 -> L2 정규화.
    이 임베딩이 '언어 무관 핵심'(공유 인코더)의 문장 표현이라, 여기다 대조
    손실을 걸어야 도메인 어댑터의 목적(용어 감각을 언어와 무관하게 학습)에
    맞는다."""
    encoder = model.get_encoder()
    enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = enc_out.last_hidden_state             
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return F.normalize(pooled, dim=-1)


def contrastive_loss(embeddings, term_ids, sense_labels, temperature=0.05):
    """SupCon 스타일: 같은 (term, sense)끼리는 가깝게, 같은 term이라도
    sense가 다르면 멀게. 배치 안의 다른 모든 예문이 분모(negative)로 쓰인다."""
    sim = embeddings @ embeddings.T / temperature          
    sim.fill_diagonal_(-1e9)

    same_term = term_ids.unsqueeze(0) == term_ids.unsqueeze(1)
    same_sense = sense_labels.unsqueeze(0) == sense_labels.unsqueeze(1)
    pos_mask = same_term & same_sense
    pos_mask.fill_diagonal_(False)

    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    log_prob = F.log_softmax(sim, dim=1)
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)
    return -pos_log_prob[has_pos].mean()


                                         

from transformers import get_linear_schedule_with_warmup

EPOCHS = 30                                          
LR = 5e-4
CONTRASTIVE_WEIGHT = 0.3                         
WARMUP_RATIO = 0.03

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
total_steps = (len(train_sampler) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
)
scaler = torch.cuda.amp.GradScaler()                                    

model.train()
global_step = 0
for epoch in range(EPOCHS):
    epoch_ce, epoch_contrastive, n_batches = 0.0, 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            ce_loss = out.loss

            emb = encoder_pooled_embeddings(model, batch["input_ids"], batch["attention_mask"])
            c_loss = contrastive_loss(emb, batch["term_id"], batch["sense_label"])

            loss = (ce_loss + CONTRASTIVE_WEIGHT * c_loss) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        epoch_ce += ce_loss.item()
        epoch_contrastive += c_loss.item()
        n_batches += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    print(f"[epoch {epoch+1}/{EPOCHS}] CE={epoch_ce/n_batches:.4f}  Contrastive={epoch_contrastive/n_batches:.4f}")

                                            
    ckpt_dir = f"{OUT_DIR}/checkpoint-epoch{epoch+1}"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  체크포인트 저장: {ckpt_dir}")

print("학습 완료")


                                                     

                                                
                                                 
                                                    

from transformers import AutoModel

LABSE_MODEL = "sentence-transformers/LaBSE"
labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL)
labse_model = AutoModel.from_pretrained(LABSE_MODEL).to(device)
labse_model.eval()


@torch.no_grad()
def labse_embed(texts, batch_size=16):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = labse_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        out = labse_model(**enc)
        emb = F.normalize(out.pooler_output, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


@torch.no_grad()
def generate_translation(model, ko_sentence, lang, use_adapter=True):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(ko_sentence, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(lang)
    if use_adapter:
        out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_LENGTH, num_beams=4)
    else:
        with model.disable_adapter():
            out_ids = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_LENGTH, num_beams=4)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]


def evaluate_sense_accuracy(model, eval_df, use_adapter, desc=""):
    """test/val 세트의 각 행에 대해: 모델이 생성한 번역이 정답(같은 term+sense
    의 gold 번역들) 쪽에 더 가까운지, 반대 sense 쪽에 더 가까운지 비교.
    pos행 정확도 = '용어 정확도'(금융 의미로 제대로 번역했는가)
    neg행 정확도 = '비금융 문장이 안 망가졌는가'(일상 의미를 유지했는가)
    """
    results = []
    for term in eval_df["term"].unique():
        term_rows = eval_df[eval_df["term"] == term]
        gold_pos = term_rows[term_rows["label"] == "pos"]["translation"].tolist()
        gold_neg = term_rows[term_rows["label"] == "neg"]["translation"].tolist()
        if not gold_pos or not gold_neg:
            continue
        pos_ref_emb = labse_embed(gold_pos).mean(dim=0, keepdim=True)
        neg_ref_emb = labse_embed(gold_neg).mean(dim=0, keepdim=True)

        for _, row in term_rows.iterrows():
            generated = generate_translation(model, row["ko_sentence"], row["lang"], use_adapter=use_adapter)
            gen_emb = labse_embed([generated])
            sim_pos = (gen_emb @ pos_ref_emb.T).item()
            sim_neg = (gen_emb @ neg_ref_emb.T).item()
            predicted_sense = "pos" if sim_pos > sim_neg else "neg"
            correct = predicted_sense == row["label"]
            results.append(
                {
                    "term": term,
                    "lang": row["lang"],
                    "true_label": row["label"],
                    "generated": generated,
                    "sim_pos": sim_pos,
                    "sim_neg": sim_neg,
                    "correct": correct,
                }
            )

    res_df = pd.DataFrame(results)
    overall_acc = res_df["correct"].mean()
    pos_acc = res_df[res_df["true_label"] == "pos"]["correct"].mean()
    neg_acc = res_df[res_df["true_label"] == "neg"]["correct"].mean()
    print(f"[{desc}] 전체 정확도={overall_acc:.1%}  용어(금융)정확도={pos_acc:.1%}  비금융유지율={neg_acc:.1%}")
    return res_df, overall_acc, pos_acc, neg_acc


                                      

model.eval()
print("=== 학습 전(순정 NLLB) — test 세트(청산/사모/모집, train에 없던 용어) ===")
base_res, base_overall, base_pos, base_neg = evaluate_sense_accuracy(model, test_df, use_adapter=False, desc="baseline")


                              

model.eval()
print("=== 학습 후(도메인 어댑터) — test 세트 ===")
after_res, after_overall, after_pos, after_neg = evaluate_sense_accuracy(model, test_df, use_adapter=True, desc="adapter")

print("\n=== Before vs After ===")
print(f"전체 정확도      : {base_overall:.1%} -> {after_overall:.1%}  ({after_overall-base_overall:+.1%})")
print(f"용어(금융) 정확도 : {base_pos:.1%} -> {after_pos:.1%}  ({after_pos-base_pos:+.1%})")
print(f"비금융 유지율     : {base_neg:.1%} -> {after_neg:.1%}  ({after_neg-base_neg:+.1%})")

after_res.to_csv(f"{OUT_DIR}/test_eval_after.csv", index=False, encoding="utf-8-sig")
base_res.to_csv(f"{OUT_DIR}/test_eval_before.csv", index=False, encoding="utf-8-sig")
print(f"\n상세 결과 저장: {OUT_DIR}/test_eval_before.csv, test_eval_after.csv")


                                

FINAL_DIR = f"{OUT_DIR}/domain_adapter_final"
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"최종 도메인 어댑터 저장: {FINAL_DIR}")

           
                                                
                                                
                                         
                                                      
                                                                  
                                                              
                                
