import pandas as pd
from pathlib import Path

input_path = Path("merged_Dataset_Yu.csv")
output_path = Path("merged_Dataset_Yu_undersampled.csv")

df = pd.read_csv(input_path)
label_col = "CommentsAssociatedLabel"

df_satd = df[df[label_col] == 1]
df_non_satd = df[df[label_col] == 0]

print("Original Distribution:")
print(df[label_col].value_counts(), "\n")

# === Undersampling ===
df_non_satd_down = df_non_satd.sample(
    n=len(df_satd),
    random_state=42
)

df_balanced = pd.concat([df_satd, df_non_satd_down]) \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

print("Distribution after undersampling")
print(df_balanced[label_col].value_counts(), "\n")


df_balanced.to_csv(output_path, index=False)
print(f"Balanced dataset saved in: {output_path}")
