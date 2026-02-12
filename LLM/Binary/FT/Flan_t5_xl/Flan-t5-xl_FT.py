import gc
import time
import torch
import pandas as pd
import numpy as np
import os
import json
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    DataCollatorWithPadding,
)

from codecarbon import EmissionsTracker






SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Dataset_Yu
# -----------------------------
df = pd.read_csv("merged_Dataset_Yu_undersampling.csv")

df = df.rename(columns={
    "CommentsAssociated": "comment_text",
    "CommentsAssociatedLabel": "label"
})

df["comment_text"] = df["comment_text"].astype(str).fillna("")
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print("Label distribution:\n", df["label"].value_counts())



# =================
#  80/20 split
# =================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
)

dataset = {
    "YuDataset": DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "valid": Dataset.from_pandas(test_df.reset_index(drop=True)),
        "test":  Dataset.from_pandas(test_df.reset_index(drop=True))
    })
}

DATASET = "YuDataset"
TEXT_COLUMN = "comment_text"
LABEL_COLUMN = "label"
NUM_LABELS = 2
METRIC = "f1"


# -----------------------------
checkpoint = "google/flan-t5-xl"
MAX_LEN = 128
BATCH_SIZE = 4
LR = 2e-5
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len = MAX_LEN



# =================
# Prompt
# =================
PROMPT = """Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.
SATD comments usually contains specific keywords: TODO, FIXME, HACK, and XXX.
Assign the label of SATD or Not-SATD for each given source code comment.
Comment: {comment}
Label:"""

def int_to_text(y):
    return "SATD" if int(y) == 1 else "Not-SATD"

def text_to_int(t):
    t = t.lower()
    if "satd" in t and "not" not in t:
        return 1
    return 0

def normalize_prediction(t):
    t = t.strip().lower()
    if "not" in t and "satd" in t:
        return "Not-SATD"
    if "satd" in t:
        return "SATD"
    return "Not-SATD"

LABEL_MAX_LEN = 8

def tokenize(batch):
    prompted = [PROMPT.format(comment=c) for c in batch[TEXT_COLUMN]]
    inputs = tokenizer(prompted, truncation=True, max_length=MAX_LEN)

    label_text = [int_to_text(y) for y in batch[LABEL_COLUMN]]
    labels = tokenizer(
        label_text,
        truncation=True,
        max_length=LABEL_MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )

    inputs["labels"] = labels["input_ids"]
    inputs["label_num"] = batch[LABEL_COLUMN]
    return inputs

# =================
# LoRA
# =================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

metric = evaluate.load(METRIC)

t0_total = time.perf_counter()
tracker = EmissionsTracker()
tracker.start()
# =================
for project_name, data in dataset.items():
    print("\n==========", project_name, "==========")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        config=AutoConfig.from_pretrained(checkpoint)
    )

    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=LR)

    tokenized = data.map(tokenize, batched=True)
    tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "label_num"]
    )

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    test_loader = DataLoader(
        tokenized["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    all_real, all_pred = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            epoch_loss += loss.item()
            num_batches += 1
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if (batch_idx + 1) % 500 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        print(f"  Epoch Loss (avg): {epoch_loss / num_batches:.4f}")

        # =================
        # Evaluation
        # =================
        print("  Evaluating on test set...")
        model.eval()
        sample_count = 0
        for batch_idx, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                gen = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=LABEL_MAX_LEN
                )

            preds_raw = tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            if batch_idx == 0 and sample_count == 0:
                refs_debug = batch["label_num"].cpu().tolist()
                print("    Sample predictions from batch 0:")
                for i in range(min(3, len(preds_raw))):
                    print(f"      Raw: '{preds_raw[i]}' | True: {int_to_text(refs_debug[i])}")
            
            preds = [normalize_prediction(p) for p in preds_raw]
            preds = [text_to_int(p) for p in preds]

            refs = batch["label_num"].cpu().tolist()

            metric.add_batch(predictions=preds, references=refs)

            if epoch == num_epochs - 1:
                all_real.extend(refs)
                all_pred.extend(preds)

        f1 = metric.compute()["f1"]
        print(f"  F1: {f1:.4f}")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

emissions = tracker.stop()
print(emissions)

print("\n===== FINAL REPORT =====")
print(f"\nPrediction distribution:")
print(f"  Class 0: {all_pred.count(0)} / {len(all_pred)}")
print(f"  Class 1: {all_pred.count(1)} / {len(all_pred)}")
print(f"\nTrue distribution:")
print(f"  Class 0: {all_real.count(0)} / {len(all_real)}")
print(f"  Class 1: {all_real.count(1)} / {len(all_real)}")

rep = classification_report(all_real, all_pred, digits=3, output_dict=True)
cm = confusion_matrix(all_real, all_pred)

print(classification_report(all_real, all_pred, digits=3))
print(cm)

os.makedirs("results_all", exist_ok=True)

report_df = pd.DataFrame(rep).transpose()
report_path = "results_all/Flan_t5_xl_FT_report_8epoch.csv"
report_df.to_csv(report_path, index=True)

cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_path = "results_all/Flan_t5_xl_FT_confusion_matrix_8epoch.csv"
cm_df.to_csv(cm_path, index=True)

print(f"\nSaved:\n- {report_path}\n- {cm_path}")



# Predictions CSV
predictions_df = pd.DataFrame({
    "real_label": all_real,
    "predicted_label": all_pred
})

predictions_path = "results_all/Flan_t5_xl_FT_predictions_8epoch.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"Saved predictions file:\n- {predictions_path}")


# Execution time
t1_total = time.perf_counter()
total_sec = t1_total - t0_total

hours = total_sec / 3600
minutes = total_sec / 60

h = int(total_sec // 3600)
m = int((total_sec % 3600) // 60)
s = int(total_sec % 60)

os.makedirs("results_all", exist_ok=True)
time_path = "results_all/total_execution_time.txt"

with open(time_path, "w", encoding="utf-8") as f:
    f.write(f"TOTAL_SECONDS: {total_sec:.2f}\n")
    f.write(f"TOTAL_MINUTES: {minutes:.2f}\n")
    f.write(f"TOTAL_HOURS: {hours:.2f}\n")
    f.write(f"FORMAT: {h}h {m}m {s}s\n")

print(f"\n[Time] Total execution time:")
print(f"  {total_sec:.2f} seconds")
print(f"  {minutes:.2f} minutes")
print(f"  {hours:.2f} hours")
print(f"  Format: {h}h {m}m {s}s")
print(f"[Time] Saved in: {time_path}")
