import os
import re
import gc
import random
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import time
from codecarbon import OfflineEmissionsTracker

t0_total = time.perf_counter()

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available())
print("device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Maldonado
# -----------------------------
DATASET = "Maldonado"
INPUT = "ct"

df = pd.read_csv("maldonado.csv")

# se vuoi essere robusto a header/spazi
df.columns = df.columns.str.strip()

df["comment_text"] = df["comment_text"].astype(str).fillna("")
df["satd"] = df["satd"].astype(int)

df["satd_str"] = df["satd"].apply(lambda x: "SATD" if int(x) == 1 else "Not-SATD")

df = add_context(df)

# tieni solo ciò che serve per l'inferenza
df = df[["context", "prompt_context", "satd_str"]].reset_index(drop=True)
df = df.rename(columns={"satd_str": "label"})



# -----------------------------
#  model/tokenizer
# -----------------------------
checkpoint = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

generation_config = GenerationConfig(max_new_tokens=5, do_sample=True, temperature=0.01)

# -----------------------------
def get_response(model, tokenizer, generation_config, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )[0],
        skip_special_tokens=True,
    )
    return output

init_prompt_for_Dataset_Maldonado_MAT = """
Self-admitted technical debt (SATD) is technical debt admitted by the developer through source code comments.
SATD comments usually contains specific keywords: TODO, FIXME, HACK, and XXX.
Assign the label of SATD or Not-SATD for each given source code comment.
""".strip()


def generate_prompt_without_adding_dynamic_examples(init_prompt, test_context):
    prompt = init_prompt + "\n\n"
    prompt += test_context + "\n"
    prompt += "### Label: "
    return prompt


# -----------------------------
def get_confmat_str(real, pred, labels):
    cm = confusion_matrix(real, pred, labels=labels)
    max_label_length = max([len(label) for label in labels] + [5])
    output = " " * max_label_length + " " + " ".join(label.ljust(max_label_length) for label in labels) + "\n"
    for i, label in enumerate(labels):
        row = " ".join([str(cm[i][j]).ljust(max_label_length) for j in range(len(labels))])
        output += label.ljust(max_label_length) + " " + row + "\n"
    return output



os.makedirs("Adding_Custom_Layers_Results", exist_ok=True)

labels = sorted(list(set(df["label"]))) 

INIT_PROMPT = init_prompt_for_Dataset_Maldonado_MAT

icl_name = "_TASKLEVEL_NOSPLIT_MAT"
file_name = f"Adding_Custom_Layers_Results/{DATASET}_Input-{INPUT}_{checkpoint.split('/')[-1]}{icl_name}"

random.seed(42)
torch.cuda.empty_cache()
gc.collect()

all_real = []
all_pred = []
all_context = []
unrecognized_pred = 0

print("Starting inference loop...")
print(f"Total rows to process: {len(df)}")


tracker = OfflineEmissionsTracker(
    country_iso_code="ITA", project_name="LLM", 
    experiment_id="Binary_Flan-t5-xl_NO_FT_Dataset_Maldonado_All", on_csv_write="append")
tracker.start()


for i, row in df.iterrows():
    if i == 0:
        print("Entered inference loop (first row)")
    if i % 1000 == 0:
        print(f"Processed {i} rows")

    prompt = generate_prompt_without_adding_dynamic_examples(init_prompt_for_Dataset_Maldonado_MAT, row["prompt_context"])
    pred = get_response(model, tokenizer, generation_config, prompt)

    for lab in labels:
        if len(pred) > 0 and pred.split()[0].lower() == lab.lower():
            pred = lab

    if pred not in labels:
        pred = "Not-SATD"
        unrecognized_pred += 1

    all_real.append(row["label"])
    all_pred.append(pred)
    all_context.append(row["prompt_context"])

emissions = tracker.stop()
print(emissions)

report = classification_report(all_real, all_pred, zero_division=0, digits=3)
confmat_str = get_confmat_str(all_real, all_pred, labels=labels)

with open(file_name + "_confmat.txt", "w") as output_file:
    output_file.write(f"Dataset: maldonado.csv\n")
    output_file.write(f"Model: {checkpoint}\n")
    output_file.write("Prompt: MAT (TODO/FIXME/HACK/XXX), task-level (no examples)\n")
    output_file.write("Evaluation: whole dataset (no split)\n\n")
    output_file.write(report + "\n")
    output_file.write(confmat_str + "\n")
    
print(report)
print(confmat_str)
print(f"\nNumber of unrecognized predictions: {unrecognized_pred}")

# Predictions CSV
orig_df = pd.read_csv("maldonado.csv")
orig_df["predicted_label"] = all_pred
full_out_path = file_name + "_full_with_predictions.csv"
orig_df.to_csv(full_out_path, index=False)
print(f"Saved full dataset with predictions:\n- {full_out_path}")