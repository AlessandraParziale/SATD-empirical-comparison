import os
import gc
import random
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available())
print("device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Dataset_Yu
# -----------------------------
DATASET = "Dataset_Yu"
INPUT = "ct"

df = pd.read_csv("merged_Dataset_Yu_undersampling.csv")

df = df.rename(
    columns={
        "CommentsAssociated": "comment_text",
        "CommentsAssociatedLabel": "satd",
    }
)

df["comment_text"] = df["comment_text"].astype(str).fillna("")
df["satd"] = df["satd"].astype(int)

df["satd_str"] = df["satd"].apply(lambda x: "SATD" if int(x) == 1 else "Not-SATD")

df = add_context(df)

df = df[["context", "prompt_context", "satd_str"]].reset_index(drop=True)
df = df.rename(columns={"satd_str": "label"})


# -----------------------------
#  model/tokenizer
# -----------------------------
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
    do_sample=True,
    temperature=0.01,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


# -----------------------------
def get_response(model, tokenizer, generation_config, prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
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


init_prompt_for_Dataset_Yu_MAT = """
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


# =============================

os.makedirs("Adding_Custom_Layers_Results", exist_ok=True)

labels = sorted(list(set(df["label"])))

INIT_PROMPT = init_prompt_for_Dataset_Yu_MAT

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

with open(file_name + "_confmat.txt", "w", encoding="utf-8") as output_file:
    output_file.write(f"Dataset: merged_Dataset_Yu_undersampling.csv\n")
    output_file.write(f"Model: {checkpoint}\n")
    output_file.write("Prompt: MAT (TODO/FIXME/HACK/XXX), task-level (no examples)\n")
    output_file.write("Evaluation: whole dataset (no split)\n\n")

    for i, row in df.iterrows():
        if i == 0:
            print("Entered inference loop (first row)")
        if i % 1000 == 0:
            print(f"Processed {i} rows")

        prompt = generate_prompt_without_adding_dynamic_examples(INIT_PROMPT, row["prompt_context"])
        pred = get_response(model, tokenizer, generation_config, prompt)

        
        for label in labels:
            if len(pred) > 0 and pred.split()[0].lower() == label.lower():
                pred = label

        if pred not in labels:
            pred = "Not-SATD"
            unrecognized_pred += 1

        all_real.append(row["label"])
        all_pred.append(pred)
        all_context.append(row["prompt_context"])

    report = classification_report(all_real, all_pred, zero_division=0, digits=3)
    print(report)
    output_file.write(report + "\n")

    confmat_str = get_confmat_str(all_real, all_pred, labels=labels)
    print(confmat_str)
    output_file.write(confmat_str + "\n")

    msg = (
        f"\nNumber of unrecognized predictions: {unrecognized_pred}\n"
        "We considered them as the majority class (Not-SATD).\n"
    )
    print(msg)
    output_file.write(msg)


test_result_df = pd.DataFrame(
    {
        "context": all_context,
        "real": all_real,
        "pred": all_pred,
    }
)
test_result_df.to_csv(file_name + "_pred.csv", index=False, encoding="utf-8")

print(f"Saved: {file_name}_pred.csv")
print(f"Saved: {file_name}_confmat.txt")
