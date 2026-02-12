import gc
import time
import torch
import pandas as pd
import numpy as np
import os
import json
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoConfig,
    DataCollatorForLanguageModeling,
)


t0_total = time.perf_counter()


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


checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
MAX_LEN = 512 
BATCH_SIZE = 2 
LR = 1e-4  
num_epochs = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 


PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.<|eot_id|><|start_header_id|>user<|end_header_id|>

SATD comments usually contain specific keywords such as TODO, FIXME, HACK, and XXX.
Assign the label SATD or Not-SATD to the following source code comment.

Comment: {comment}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""



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


def tokenize(batch):
    prompted = [PROMPT_TEMPLATE.format(comment=c) for c in batch[TEXT_COLUMN]]
    label_text = [int_to_text(y) for y in batch[LABEL_COLUMN]]
    
    full_text = [p + l + tokenizer.eos_token for p, l in zip(prompted, label_text)]
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    
    labels = tokenized["input_ids"].clone()
    
    for i, (prompt, label) in enumerate(zip(prompted, label_text)):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
    
        labels[i, :prompt_len] = -100
        
        labels[i, labels[i] == tokenizer.pad_token_id] = -100
    
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "label_num": batch[LABEL_COLUMN]
    }
    
    return result

# =================
# LoRA
# =================
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM 
)

metric = evaluate.load(METRIC)


for project_name, data in dataset.items():
    print("\n==========", project_name, "==========")
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,  
        device_map="auto", 
        trust_remote_code=True
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Tokenizing dataset...")
    tokenized = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False 
    )

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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch Loss (avg): {avg_loss:.4f}")

        # =================
        # Evaluation
        # =================
        print("  Evaluating on test set...")
        model.eval()
        
        test_data = data["test"]
        test_prompts = [PROMPT_TEMPLATE.format(comment=c) for c in test_data[TEXT_COLUMN]]
        test_labels = test_data[LABEL_COLUMN]
        
        batch_preds = []
        batch_refs = []
        
        for i in range(0, len(test_prompts), BATCH_SIZE):
            batch_prompts = test_prompts[i:i+BATCH_SIZE]
            batch_labels_true = test_labels[i:i+BATCH_SIZE]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(device)
            
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=20, 
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False, 
                    temperature=None,
                    top_p=None
                )
            
            input_len = inputs["input_ids"].shape[1]
            preds_raw = tokenizer.batch_decode(
                gen[:, input_len:],
                skip_special_tokens=True
            )
            
            if i == 0:
                print("    Sample predictions:")
                for j in range(min(3, len(preds_raw))):
                    true_label = int_to_text(batch_labels_true[j])
                    print(f"      Raw: '{preds_raw[j][:50]}' | True: {true_label}")
            
            preds = [normalize_prediction(p) for p in preds_raw]
            preds = [text_to_int(p) for p in preds]
            
            batch_preds.extend(preds)
            batch_refs.extend(batch_labels_true)
        
        metric.add_batch(predictions=batch_preds, references=batch_refs)
        f1 = metric.compute()["f1"]
        print(f"  F1: {f1:.4f}")
        
        if epoch == num_epochs - 1:
            all_real = batch_refs
            all_pred = batch_preds
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

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

os.makedirs("results_all_LLAMA", exist_ok=True)
report_df = pd.DataFrame(rep).transpose()
report_path = "results_all_LLAMA/Llama_3.1_8B_FT_report_8epoch.csv"
report_df.to_csv(report_path, index=True)

cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_path = "results_all_LLAMA/Llama_3.1_8B_FT_confusion_matrix_8epoch.csv"
cm_df.to_csv(cm_path, index=True)

# Ppredictions CSV
predictions_df = pd.DataFrame({
    "real_label": all_real,
    "predicted_label": all_pred
})

predictions_path = "results_all_LLAMA/Llama_3.1_8B_FT_predictions_8epoch.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"- Predictions: {predictions_path}")

model_save_path = "results_all_LLAMA/llama_3.1_8b_satd_lora"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"\nSaved:")
print(f"- {report_path}")
print(f"- {cm_path}")
print(f"- Model: {model_save_path}")


# Execution time
t1_total = time.perf_counter()
total_sec = t1_total - t0_total

hours = total_sec / 3600
minutes = total_sec / 60

h = int(total_sec // 3600)
m = int((total_sec % 3600) // 60)
s = int(total_sec % 60)

os.makedirs("results_all_LLAMA", exist_ok=True)
time_path = os.path.join("results_all_LLAMA", "total_execution_time.txt")

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
