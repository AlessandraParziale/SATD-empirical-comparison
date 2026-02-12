import os
import gc
import random

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


# -----------------------------
def add_context(df):
    context = []
    prompt_context = []
    for _, row in df.iterrows():
        context.append(row["comment_text"])
        prompt_context.append('### Technical debt comment: """ ' + row["comment_text"] + ' """')
    df["context"] = context
    df["prompt_context"] = prompt_context
    return df


# -----------------------------
# device
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

df = df.rename(
    columns={
        "comment_text": "comment_text",
        "classification": "label",
    }
)

if "comment_text" not in df.columns:
    raise ValueError("Missing column: comment_text")
if "label" not in df.columns and "classification" in df.columns:
    df = df.rename(columns={"classification": "label"})
if "label" not in df.columns:
    raise ValueError("Missing column: label/classification")

if "satd" not in df.columns:
    raise ValueError("Missing column: satd")
df["satd"] = pd.to_numeric(df["satd"], errors="coerce")
df = df[df["satd"] == 1].reset_index(drop=True)

df["comment_text"] = df["comment_text"].astype(str).fillna("")


labels = ["IMPLEMENTATION", "TEST", "DEFECT", "DESIGN", "DOCUMENTATION"]
MAJORITY_CLASS = "IMPLEMENTATION"

df["label"] = df["label"].astype(str).str.strip().str.upper()
df = df[df["label"].isin(labels)].reset_index(drop=True)

df = add_context(df)
df = df[["context", "prompt_context", "label"]].reset_index(drop=True)

# =================
#  Model
# =================
checkpoint = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

generation_config = GenerationConfig(max_new_tokens=5, do_sample=False, temperature=0.0)

# =================
# Prompt
# =================
def get_response(model, tokenizer, generation_config, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = tokenizer.decode(
        model.generate(inputs["input_ids"], generation_config=generation_config)[0],
        skip_special_tokens=True,
    )
    return output


init_prompt_for_SATD_Classification = """
Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.
There are five types of software technical debts:

Implementation: Bad coding practices leading to poor legibility of code, making it difficult to understand and maintain.
Test: Problems found in implementations involving testing or monitoring subcomponents.
Defect: Identified defects in the system that should be addressed.
Design: Areas which violate good software design practices, causing poor flexibility to evolving business needs.
Documentation: Inadequate documentation that exists within the software system. 

Choose and assign ONE label among: Implementation, Test, Defect, Design, or Documentation. 
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



os.makedirs("SATD_Classification_Results_Flan", exist_ok=True)

INIT_PROMPT = init_prompt_for_SATD_Classification
icl_name = "_CLASSIFICATION_NOSPLIT_ZEROSHOT"
file_name = f"SATD_Classification_Results_Flan/{DATASET}_Input-{INPUT}_{checkpoint.split('/')[-1]}{icl_name}"

random.seed(42)
torch.cuda.empty_cache()
gc.collect()

all_real, all_pred, all_context = [], [], []
unrecognized_pred = 0

print("Starting inference loop...")
print(f"Total rows to process: {len(df)}")

with open(file_name + "_confmat.txt", "w") as output_file:
    output_file.write("Evaluation:\n\n")

    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"Processed {i}/{len(df)}")

        prompt = generate_prompt_without_adding_dynamic_examples(INIT_PROMPT, row["prompt_context"])
        pred_raw = get_response(model, tokenizer, generation_config, prompt)

        pred_token = pred_raw.strip().split()[0].upper() if str(pred_raw).strip() else ""
        if pred_token not in labels:
            pred_token = MAJORITY_CLASS
            unrecognized_pred += 1

        all_real.append(row["label"])
        all_pred.append(pred_token)
        all_context.append(row["prompt_context"])

    report = classification_report(all_real, all_pred, labels=labels, zero_division=0, digits=3)
    print(report)
    output_file.write(report + "\n")

    confmat_str = get_confmat_str(all_real, all_pred, labels=labels)
    print(confmat_str)
    output_file.write(confmat_str + "\n")

    msg = f"\nNumber of unrecognized predictions: {unrecognized_pred}\n"
    print(msg)
    output_file.write(msg)

pd.DataFrame({"context": all_context, "real": all_real, "pred": all_pred}).to_csv(file_name + "_pred.csv", index=False)
print(f"Saved: {file_name}_pred.csv")
print(f"Saved: {file_name}_confmat.txt")
