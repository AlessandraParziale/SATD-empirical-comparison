import os, gc, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorWithPadding,
    get_scheduler,
)



HF_TOKEN = "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"  
checkpoint = "meta-llama/Llama-3.1-8B-Instruct"


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] cuda:", torch.cuda.is_available(), "| device:", device)
if torch.cuda.is_available():
    print("[INFO] GPU:", torch.cuda.get_device_name(0))



# =========================
PROMPT_PREFIX = """
Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.
There are five types of software technical debts:

Implementation: Bad coding practices leading to poor legibility of code, making it difficult to understand and maintain.
Test: Problems found in implementations involving testing or monitoring subcomponents.
Defect: Identified defects in the system that should be addressed.
Design: Areas which violate good software design practices, causing poor flexibility to evolving business needs.
Documentation: Inadequate documentation that exists within the software system.

Choose and assign ONE label among: Implementation, Test, Defect, Design, or Documentation.
""".strip()

def build_input_text(comment_text: str) -> str:
    return (
        PROMPT_PREFIX
        + "\n\n"
        + '### Technical debt comment: """ '
        + str(comment_text)
        + ' """'
    )


# -----------------------------
# Dataset Maldonado
# -----------------------------
df = pd.read_csv("maldonado.csv")

df["satd"] = pd.to_numeric(df["satd"], errors="coerce")
df = df[df["satd"] == 1].reset_index(drop=True)

label_list = ["IMPLEMENTATION", "TEST", "DEFECT", "DESIGN", "DOCUMENTATION"]
df["label"] = df["classification"].astype(str).str.strip().str.upper()
df = df[df["label"].isin(label_list)].reset_index(drop=True)

df["comment_text"] = df["comment_text"].astype(str).fillna("")
df["text"] = df["comment_text"].apply(build_input_text)

label2id = {lab: i for i, lab in enumerate(label_list)}
id2label = {i: lab for lab, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

df = df[["text", "label_id"]].reset_index(drop=True)

print("[INFO] rows:", len(df))
print("[INFO] class counts:\n", df["label_id"].value_counts().sort_index())


# =========================
# Train/Test split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label_id"],
)

train_counts = train_df["label_id"].value_counts().sort_index()
class_weights = (1.0 / torch.tensor(train_counts.values, dtype=torch.float))
class_weights = class_weights / class_weights.sum() * len(class_weights)

print("[INFO] train counts:", train_counts.to_dict())
print("[INFO] class weights:", [round(x, 4) for x in class_weights.tolist()])

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
}).rename_column("label_id", "labels")

for split in ["train", "test"]:
    for col in ds[split].column_names:
        if col.startswith("__index_level_0__"):
            ds[split] = ds[split].remove_columns([col])

print("[INFO] train:", len(ds["train"]), "| test:", len(ds["test"]))



# =========================
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    token=HF_TOKEN,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



# =========================
def print_trainable_params(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")


class LlamaWithLoRAForClassification(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int, class_weights: torch.Tensor, hf_token: str):
        super().__init__()

        cfg = AutoConfig.from_pretrained(checkpoint, token=hf_token)

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.backbone = AutoModel.from_pretrained(
            checkpoint,
            config=cfg,
            token=hf_token,
            torch_dtype=dtype,
        )

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        hidden_size = cfg.hidden_size  
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state 

        mask = attention_mask.unsqueeze(-1).float() 
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6) 

        pooled = self.dropout(pooled)
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)



# =========================
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 2e-4
EPOCHS = 8

print("[INFO] checkpoint:", checkpoint)
print("[INFO] MAX_LEN:", MAX_LEN, "| EPOCHS:", EPOCHS, "| BATCH_SIZE:", BATCH_SIZE,
      "| GRAD_ACCUM:", GRAD_ACCUM_STEPS, "| LR:", LR)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

tokenized = ds.map(tokenize_fn, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
test_loader  = DataLoader(tokenized["test"],  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)


# =========================
model = LlamaWithLoRAForClassification(checkpoint, len(label_list), class_weights, HF_TOKEN).to(device)
print_trainable_params(model)

optimizer = AdamW(model.parameters(), lr=LR)

num_training_steps = EPOCHS * (len(train_loader) // GRAD_ACCUM_STEPS + 1)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

use_amp = torch.cuda.is_available()
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16


scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(1, len(train_loader))
    print(f"[TRAIN] epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f}")

    torch.cuda.empty_cache()
    gc.collect()


# =========================
# Metrics
# =========================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        pred = torch.argmax(out.logits, dim=-1).cpu().tolist()
        y_pred += pred
        y_true += batch["labels"].cpu().tolist()

y_true_lbl = [id2label[i] for i in y_true]
y_pred_lbl = [id2label[i] for i in y_pred]

report_str = classification_report(
    y_true_lbl, y_pred_lbl,
    labels=label_list,
    zero_division=0,
    digits=3
)
cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=label_list)

print("\n[TEST] classification_report")
print(report_str)
print("[TEST] confusion_matrix (rows=true, cols=pred)")
print(cm)

out_dir = "ft_satd_lora_results_llama31_8b"
os.makedirs(out_dir, exist_ok=True)

metrics_path = os.path.join(out_dir, "maldonado_llama31-8b_lora_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("[TEST] classification_report\n")
    f.write(report_str)
    f.write("\n\n[TEST] confusion_matrix (rows=true, cols=pred)\n")
    f.write(np.array2string(cm))
    f.write("\n")
print("saved metrics:", metrics_path)

cm_csv = os.path.join(out_dir, "maldonado_llama31-8b_lora_confmat.csv")
pd.DataFrame(cm, index=label_list, columns=label_list).to_csv(cm_csv, index=True)
print("saved confmat csv:", cm_csv)

pred_csv = os.path.join(out_dir, "maldonado_llama31-8b_lora_preds.csv")
pd.DataFrame({
    "text": test_df["text"].tolist(),
    "true": y_true_lbl,
    "pred": y_pred_lbl,
}).to_csv(pred_csv, index=False)
print("saved preds:", pred_csv)

lora_dir = os.path.join(out_dir, "llama31-8b_lora_adapters")
model.backbone.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print("saved LoRA adapters:", lora_dir)

