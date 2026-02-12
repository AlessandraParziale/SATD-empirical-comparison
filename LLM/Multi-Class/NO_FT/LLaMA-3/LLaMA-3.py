import os
import gc
import random
import re
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# -----------------------------
def add_context(df):
    context = []
    prompt_context = []
    for _, row in df.iterrows():
        context.append(row["comment_text"])
        prompt_context.append(
            '### Technical debt comment: """ ' + row["comment_text"] + ' """'
        )
    df["context"] = context
    df["prompt_context"] = prompt_context
    return df


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] cuda available:", torch.cuda.is_available())
print("[INFO] device:", device)
if torch.cuda.is_available():
    print("[INFO] GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Dataset Maldonado
# -----------------------------
DATASET = "maldonado"
INPUT = "ct"

df = pd.read_csv("maldonado.csv")
print("[INFO] CSV columns:", list(df.columns))

df = df.rename(columns={"classification": "label"})

if "comment_text" not in df.columns:
    raise ValueError("Missing column: comment_text")
if "label" not in df.columns:
    raise ValueError("Missing column: classification")

if "satd" not in df.columns:
    raise ValueError("Missing column: satd")
df["satd"] = pd.to_numeric(df["satd"], errors="coerce")
df = df[df["satd"] == 1].reset_index(drop=True)

df["comment_text"] = df["comment_text"].astype(str).fillna("")


labels = ["IMPLEMENTATION", "TEST", "DEFECT", "DESIGN", "DOCUMENTATION"]
MAJORITY_CLASS = "IMPLEMENTATION"

df["label"] = df["label"].astype(str).str.strip().str.upper()
df = df[df["label"].isin(labels)].reset_index(drop=True)

print("[INFO] Rows:", len(df))
print("[INFO] Label distribution:\n", df["label"].value_counts())

df = add_context(df)
df = df[["context", "prompt_context", "label"]].reset_index(drop=True)


# =================
# Model 
# ================
checkpoint = "meta-llama/Llama-3.1-8B-Instruct"

HF_TOKEN = "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"  

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    use_fast=True,
    token=HF_TOKEN
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

model.config.pad_token_id = tokenizer.pad_token_id


generation_config = GenerationConfig(
    max_new_tokens=10,   
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


# =================

def get_response(model, tokenizer, generation_config, prompt):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, generation_config=generation_config)

    gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def map_prediction_to_label(pred_raw, labels, majority_class):
    pred_up = str(pred_raw).strip().upper()
    if not pred_up:
        return majority_class, True

    for lab in labels:
        if lab in pred_up:
            return lab, False

    first = pred_up.split()[0] if pred_up.split() else ""
    first = re.sub(r"[^A-Z&]", "", first)  
    if first in labels:
        return first, False

    return majority_class, True


# =================
# Prompt
# =================

init_prompt_for_SATD_Classification = """
Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.
There are five types of software technical debts:

Implementation: Bad coding practices leading to poor legibility of code, making it difficult to understand and maintain.
Test: Problems found in implementations involving testing or monitoring subcomponents.
Defect: Identified defects in the system that should be addressed.
Design: Areas which violate good software design practices, causing poor flexibility to evolving business needs.
Documentation: Inadequate documentation that exists within the software system.

Task:
Choose and assign ONE label among: Implementation, Test, Defect, Design, Documentation.
""".strip()


def generate_prompt_without_adding_dynamic_examples(init_prompt, test_context):
    prompt = init_prompt + "\n\n"
    prompt += test_context + "\n"
    prompt += "### Label: "
    return prompt


def get_confmat_str(real, pred, labels):
    cm = confusion_matrix(real, pred, labels=labels)
    max_label_length = max([len(label) for label in labels] + [5])
    output = " " * max_label_length + " " + " ".join(label.ljust(max_label_length) for label in labels) + "\n"
    for i, label in enumerate(labels):
        row = " ".join([str(cm[i][j]).ljust(max_label_length) for j in range(len(labels))])
        output += label.ljust(max_label_length) + " " + row + "\n"
    return output


OUT_DIR = "SATD_Classification_Results_LLAMA"
os.makedirs(OUT_DIR, exist_ok=True)

INIT_PROMPT = init_prompt_for_SATD_Classification
icl_name = "_CLASSIFICATION_NOSPLIT_ZEROSHOT"
file_name = f"{OUT_DIR}/{DATASET}_Input-{INPUT}_{checkpoint.split('/')[-1]}{icl_name}"

random.seed(42)
torch.cuda.empty_cache()
gc.collect()

all_real, all_pred, all_context = [], [], []
unrecognized_pred = 0

print("Starting inference loop...")
print(f"Total rows: {len(df)}")

with open(file_name + "_confmat.txt", "w", encoding="utf-8") as output_file:
    output_file.write("Evaluation:\n\n")

    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"Processed {i}/{len(df)}")

        prompt = generate_prompt_without_adding_dynamic_examples(
            INIT_PROMPT, row["prompt_context"]
        )
        pred_raw = get_response(model, tokenizer, generation_config, prompt)

        if i < 5:
            print("RAW:", repr(pred_raw))

        pred_token, was_unrecognized = map_prediction_to_label(
            pred_raw, labels, MAJORITY_CLASS
        )
        if was_unrecognized:
            unrecognized_pred += 1

        all_real.append(row["label"])
        all_pred.append(pred_token)
        all_context.append(row["prompt_context"])

    report = classification_report(
        all_real, all_pred, labels=labels, zero_division=0, digits=3
    )
    print(report)
    output_file.write(report + "\n")

    confmat_str = get_confmat_str(all_real, all_pred, labels)
    print(confmat_str)
    output_file.write(confmat_str + "\n")

    msg = f"\nNumber of unrecognized predictions: {unrecognized_pred}\n"
    print(msg)
    output_file.write(msg)

pd.DataFrame(
    {"context": all_context, "real": all_real, "pred": all_pred}
).to_csv(file_name + "_pred.csv", index=False, encoding="utf-8")

print(f"Saved: {file_name}_pred.csv")
print(f"Saved: {file_name}_confmat.txt")
